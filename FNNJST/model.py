import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
from collections import Counter
import copy
from transformers import AutoTokenizer, AutoModel
from tqdm.notebook import tqdm

from ptdec.dec import DEC
from ptdec.model import train
import ptsdae.model as ae
from ptsdae.sdae import StackedDenoisingAutoEncoder

from FNNJST.dataset import CachedBERTDataset

try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
except ImportError:
    class DummyWriter:
        def add_scalars(self, *args, **kwargs):
            pass
    writer = DummyWriter()

class FNNJST:
    def __init__(self, texts, labels, bert_model, cuda=True, #Bert Representation
                silent_encoder_params = True ,encoders_input_params = 768, encoder_output_params = 5, encoder_epochs=100, encoder_batch_size=10, #Encoder Set UP
                sentiment_epochs=50, sentiment_learning_rate=0.001, sentiment_batch_size=10, #Sentiment Set UP
                cluster_number=5, hidden_dimension=5, dec_epochs=50, dec_batch_size=10): #DEC Set UP
        self.texts = texts
        self.labels = labels
        self.bert_model = bert_model
        self.cuda = cuda
  
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.model_dec = None 
        self.model_sentiment = None
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model)

        self.input_encoder_params = encoders_input_params
        self.output_encoder_params = encoder_output_params
        self.encoder_epochs = encoder_epochs
        self.encoder_batch_size = encoder_batch_size
        self.silent_encoder_params = silent_encoder_params

        self.model_sentiment_epochs = sentiment_epochs
        self.model_sentiment_learning_rate = sentiment_learning_rate
        self.model_sentiment_batch_size = sentiment_batch_size

        self.cluster_number = cluster_number
        self.hidden_dimension = hidden_dimension
        self.model_dec_epochs = dec_epochs
        self.model_dec_batch_size = dec_batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        
        self.class_labels = ["Negative", "Positive"]
        
        self.train_loader = None
        self.val_loader = None

        self.encoder = None
        self.decoder = None

        self.dataset = CachedBERTDataset(texts, labels, bert_model=bert_model, max_length=128, cuda=cuda, testing_mode=False)
    
    def training_callback(self, epoch, lr, loss):
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss},
            epoch,
        )
    
    def train_autoencoder(self, input_params=None, output_params=None):
        """
        Train a stacked denoising autoencoder
        
        Args:
            input_params: Input dimension for the autoencoder (defaults to self.input_encoder_params)
            output_params: Output dimension for the autoencoder (defaults to self.output_encoder_params)
        
        Returns:
            tuple: (encoder, decoder) parts of the autoencoder
        """
        if input_params is None:
            input_params = self.input_encoder_params
        if output_params is None:
            output_params = self.output_encoder_params

        autoencoder = StackedDenoisingAutoEncoder([input_params, 500, 500, 2000, output_params], final_activation=None)
        if torch.cuda.is_available() and self.cuda:
            autoencoder.cuda()
        
        print("Pretraining stage.")
        ae.pretrain(
            self.dataset,
            autoencoder,
            cuda=torch.cuda.is_available() and self.cuda,
            validation=None,
            epochs=self.encoder_epochs, 
            batch_size=self.encoder_batch_size, 
            optimizer=lambda model: SGD(model.parameters(), lr=0.01, momentum=0.9),
            scheduler=lambda x: StepLR(x, 100, gamma=0.1),
            corruption=0.2,
            silent=self.silent_encoder_params
        )

        print("Training stage.")
        ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.01, momentum=0.9)
        ae.train(
            self.dataset,
            autoencoder,
            cuda=torch.cuda.is_available() and self.cuda,
            validation=None,
            epochs=self.encoder_epochs, 
            batch_size=self.encoder_batch_size,
            optimizer=ae_optimizer,
            scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
            corruption=0.2,
            update_callback=None, 
            silent=self.silent_encoder_params
        )

        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        return self.encoder, self.decoder
    
    def prepare_data(self, dataset, train_ratio=0.8, batch_size=32):
        """
        Split the dataset into training and validation sets
        
        Args:
            dataset: Dataset containing inputs and labels
            train_ratio: Ratio of data to use for training (default: 0.8)
            batch_size: Batch size for data loaders
            
        Returns:
            train_loader, val_loader: DataLoader objects for training and validation
        """
        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        print(f"Dataset split into {train_size} training samples and {val_size} validation samples")
        return self.train_loader, self.val_loader
    
    def prepare_data_stratified(self, dataset, train_ratio=0.8, batch_size=10):
        """
        Split the dataset into training and validation sets using stratified sampling
        
        Args:
            dataset: Dataset containing inputs and labels
            train_ratio: Ratio of data to use for training (default: 0.8)
            batch_size: Batch size for data loaders
            
        Returns:
            train_loader, val_loader: DataLoader objects for training and validation
        """
        from sklearn.model_selection import train_test_split
        
        all_labels = []
        for _, label in dataset:
            if isinstance(label, torch.Tensor):
                if label.dim() > 0 and label.size(0) > 1:
                    label = torch.argmax(label).item()
                else:
                    label = label.item()
            all_labels.append(label)
        
        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=1-train_ratio,
            stratify=all_labels,
            random_state=42
        )
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        print(f"Dataset split into {len(train_indices)} training samples and {len(val_indices)} validation samples (stratified)")
        return self.train_loader, self.val_loader
    
    def train_SENTIMENT(self, learning_rate=None, num_epochs=None, batch_size=None, val_ratio=0.2, dataset=None):
        """
        Train the sentiment classifier using the autoencoder
        
        Args:
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            val_ratio: Validation set ratio
            dataset: Optional dataset to use instead of self.dataset
        
        Returns:
            training_stats: Dictionary containing training and validation metrics
        """

        learning_rate = learning_rate if learning_rate is not None else self.model_sentiment_learning_rate
        num_epochs = num_epochs if num_epochs is not None else self.model_sentiment_epochs
        batch_size = batch_size if batch_size is not None else self.model_sentiment_batch_size

        self.model_sentiment = nn.Sequential(
            self.encoder, 
            self.decoder,
            nn.Linear(768, 100),     
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)      
        )
        
        available_devices = []
        if torch.cuda.is_available() and self.cuda:
            num_gpus = torch.cuda.device_count()
            for i in range(min(num_gpus, 2)): 
                available_devices.append(torch.device(f"cuda:{i}"))
        
        if not available_devices:
            available_devices = [self.device]  

        self.model_sentiment = self.model_sentiment.to(available_devices[0])
        
        for module in self.model_sentiment:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        dataset_to_use = dataset if dataset is not None else self.dataset
        if self.train_loader is None:
            try:
                self.prepare_data_stratified(dataset_to_use, train_ratio=1-val_ratio, batch_size=batch_size)
            except:
                self.prepare_data(dataset_to_use, train_ratio=1-val_ratio, batch_size=batch_size)
        

        self.optimizer = AdamW(self.model_sentiment.parameters(), lr=learning_rate)
        
        training_stats = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        epoch_pbar = tqdm(range(num_epochs), desc="Training Epochs", position=0)
        
        for epoch in epoch_pbar:
            self.model_sentiment.train()
            
            total_train_loss = 0
            batch_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", position=1, leave=False)
            
            for batch in batch_pbar:
                inputs, labels = batch  
                inputs = inputs.to(available_devices[0])
                
                labels = labels.long().to(available_devices[0])
                
                if labels.dim() > 1 and labels.shape[1] > 1:
                    labels = torch.argmax(labels, dim=1)
                    
                self.optimizer.zero_grad() 
                
                outputs = self.model_sentiment(inputs)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                batch_loss = loss.item()
                total_train_loss += batch_loss
                
                batch_pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            training_stats['train_loss'].append(avg_train_loss)
            
            epoch_stats = {"train_loss": f"{avg_train_loss:.4f}"}
            
            if self.val_loader is not None:
                val_pbar = tqdm(desc="Validating", position=1, leave=False, total=1)
                val_loss, val_accuracy = self._validate_sentiment()
                val_pbar.update(1)
                val_pbar.close()
                
                training_stats['val_loss'].append(val_loss)
                training_stats['val_accuracy'].append(val_accuracy)
                
                epoch_stats.update({
                    "val_loss": f"{val_loss:.4f}", 
                    "val_accuracy": f"{val_accuracy:.4f}"
                })
            
            epoch_pbar.set_postfix(epoch_stats)
            
        print("Sentiment Training complete!")
        return training_stats
    
    def train_DEC(self, cluster_number=None, hidden_dimension=None, batch_size=None, epochs=None):
        """
        Train the DEC model using the encoder
        
        Args:
            cluster_number: Number of clusters (optional)
            hidden_dimension: Dimension of the hidden representation (optional)
        
        Returns:
            model_dec: Trained DEC model
        """
        if cluster_number is None:
            cluster_number = self.cluster_number
        if hidden_dimension is None:
            hidden_dimension = self.hidden_dimension
        if batch_size is None:
            batch_size = self.model_dec_batch_size
        if epochs is None:
            epochs = self.model_dec_epochs
        
        available_devices = []
        if torch.cuda.is_available() and self.cuda:
            num_gpus = torch.cuda.device_count()
            for i in range(min(num_gpus, 2)): 
                available_devices.append(torch.device(f"cuda:{i}"))
        
        if not available_devices:
            available_devices = [self.device] 
        
        dec_device = available_devices[1] if len(available_devices) > 1 else available_devices[0]
     
        encoder_copy = copy.deepcopy(self.encoder)
        encoder_copy = encoder_copy.to(dec_device)
        
        self.model_dec = DEC(cluster_number=cluster_number, hidden_dimension=hidden_dimension, encoder=encoder_copy)
        self.model_dec = self.model_dec.to(dec_device)
        
        dec_optimizer = SGD(self.model_dec.parameters(), lr=0.01, momentum=0.9)
        
        dec_loader = DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=True
        )
        
        train(
            dataset=self.dataset,
            model=self.model_dec,
            epochs=self.model_dec_epochs,
            batch_size=self.model_dec_batch_size,
            optimizer=dec_optimizer,
            stopping_delta=0.00000001,
            cuda=True if "cuda" in str(dec_device) else False,
        )
        
        return self.model_dec
    
    def _validate_sentiment(self):
        """
        Run validation on the validation set
        
        Returns:
            val_loss: Average validation loss
            val_accuracy: Validation accuracy
        """
        self.model_sentiment.eval()  
        total_val_loss = 0
        correct = 0
        total = 0
        
        device = next(self.model_sentiment.parameters()).device
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(device)
                labels = labels.long().to(device)
                
                if labels.dim() > 1 and labels.shape[1] > 1:
                    labels = torch.argmax(labels, dim=1)
                
                outputs = self.model_sentiment(inputs)
                
                loss = self.criterion(outputs, labels)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = total_val_loss / len(self.val_loader)
        val_accuracy = correct / total
        
        return val_loss, val_accuracy
    
    def evaluate(self, test_dataset=None, batch_size=32):
        """
        Evaluate the sentiment classifier
        
        Args:
            test_dataset: PyTorch Dataset containing inputs and labels (optional)
            batch_size: Batch size for evaluation
            
        Returns:
            accuracy: Overall accuracy
            metrics: Dictionary with precision, recall, and f1-score
        """
        if test_dataset is None and self.val_loader is not None:
            test_loader = self.val_loader
        elif test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            raise ValueError("No test data provided. Either provide test_dataset or prepare data first.")
        
        self.model_sentiment.eval()  
        
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        device = next(self.model_sentiment.parameters()).device
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.long().to(device)
                
                if labels.dim() > 1 and labels.shape[1] > 1:
                    labels = torch.argmax(labels, dim=1)
                
                outputs = self.model_sentiment(inputs)
                
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f}")
        
        try:
            from sklearn.metrics import classification_report, precision_recall_fscore_support
            metrics = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
            metrics = {
                'precision': metrics[0],
                'recall': metrics[1],
                'f1': metrics[2]
            }
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            
            print("\nDetailed Classification Report:")
            print(classification_report(all_labels, all_predictions, target_names=self.class_labels))
            
        except ImportError:
            metrics = None
            print("Install scikit-learn for detailed metrics")
        
        return accuracy, metrics
    
    def predict(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]
    
        if isinstance(inputs, list) and isinstance(inputs[0], str):
            tokens = self.bert_tokenizer(
                inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)
    
            with torch.no_grad():
                if not callable(self.bert_model):
                    from transformers import AutoModel
                    self.bert_model = AutoModel.from_pretrained(self.bert_model if isinstance(self.bert_model, str) else "indolem/indobert-base-uncased")
                    self.bert_model.to(self.device)
                
                outputs = self.bert_model(**tokens)
            
            embeddings_tensor = outputs.last_hidden_state[:, 0, :]
    
        elif isinstance(inputs, torch.Tensor):
            embeddings_tensor = inputs
        else:
            raise ValueError("Input must be a list of texts or embeddings tensor")
    
        sentiment_device = next(self.model_sentiment.parameters()).device
        dec_device = next(self.model_dec.parameters()).device
        
        self.model_sentiment.eval()
        self.model_dec.eval()
        
        with torch.no_grad():
            sentiment_inputs = embeddings_tensor.to(sentiment_device)
            sentiment_inputs =  ae.predict(dataset = sentiment_inputs, 
                                           model = self.decoder,
                                           batch_size =  self.model_sentiment_batch_size ,
                                           silent=True
                                           )
            sentiment_outputs = self.model_sentiment(sentiment_inputs)
            sentiment_probs, sentiment_preds = F.softmax(sentiment_outputs, dim=1).max(dim=1)
            
            dec_inputs = embeddings_tensor.to(dec_device)
            dec_inputs =  ae.predict(dataset = dec_inputs,
                                     model =self.encoder,
                                     batch_size = self.model_dec_batch_size,
                                     silent=True
                                        )
            cluster_outputs = self.model_dec(dec_inputs)
            cluster_preds = cluster_outputs.argmax(dim=1)
        
        sentiment_predictions = sentiment_preds.cpu().numpy()
        sentiment_probabilities = sentiment_probs.cpu().numpy()
        cluster_predictions = cluster_preds.cpu().numpy()
        
        results = []
        for i in range(len(sentiment_predictions)):
            sentiment_label = self.class_labels[sentiment_predictions[i]]
            result = {
                'sentiment': sentiment_label,
                'sentiment_probability': float(sentiment_probabilities[i]),
                'cluster': int(cluster_predictions[i])
            }
            if isinstance(inputs, list) and isinstance(inputs[0], str):
                result['text'] = inputs[i]
            results.append(result)
        
        return results
        
    def train_multi_task(self, learning_rate=0.001, sentiment_epochs=20, dec_epochs=50, batch_size=2, val_ratio=0.2, cluster_number=None, hidden_dimension=None):
        """
        Train both sentiment and DEC models in parallel if multiple GPUs are available,
        or sequentially if only one GPU is available.
        
        Args:
            learning_rate: Learning rate for sentiment optimizer
            sentiment_epochs: Number of epochs for sentiment training
            dec_epochs: Number of epochs for DEC training
            batch_size: Batch size for training
            val_ratio: Validation set ratio
            cluster_number: Number of clusters for DEC
            hidden_dimension: Hidden dimension for DEC
            
        Returns:
            dict: Dictionary containing both trained models
        """
        if self.encoder is None or self.decoder is None:
            print("Training autoencoder...")
            self.encoder, self.decoder = self.train_autoencoder()

        multi_gpu = False
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and self.cuda:
            multi_gpu = True
            print(f"Found {torch.cuda.device_count()} GPUs. Training models in parallel.")
        
        if multi_gpu:
            import threading
            
            sentiment_thread = threading.Thread(
                target=self.train_SENTIMENT,
                kwargs={
                    'learning_rate': learning_rate,
                    'num_epochs': sentiment_epochs,
                    'batch_size': batch_size,
                    'val_ratio': val_ratio
                }
            )
            
            dec_thread = threading.Thread(
                target=self.train_DEC,
                kwargs={
                    'cluster_number': cluster_number,
                    'hidden_dimension': hidden_dimension
                }
            )
            
            sentiment_thread.start()
            dec_thread.start()
            
            sentiment_thread.join()
            dec_thread.join()
        else:
            print("Training sentiment classifier...")
            self.train_SENTIMENT(
            )
            
            print("Training DEC model...")
            self.train_DEC(
            )
        
        print("Multi-task training complete!")
        
        return {
            'sentiment_model': self.model_sentiment,
            'dec_model': self.model_dec
        }
    
    def get_cluster_assignments(self, model_dec=None, dataset=None, batch_size=32, cuda=True):
        if model_dec is None:
            model_dec = self.model_dec
        if dataset is None:
            dataset = self.dataset
            
        model_dec.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        cluster_assignments = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, tuple) or isinstance(batch, list):
                    features = batch[0]
                else:
                    features = batch
                    
                if cuda:
                    features = features.cuda()
                
                output = model_dec(features)
                cluster_pred = output.argmax(dim=1)
                
                cluster_assignments.extend(cluster_pred.cpu().numpy())
                
        return np.array(cluster_assignments)

    def map_texts_to_clusters(self, texts, cluster_assignments):
        clusters = {}
        
        n = min(len(texts), len(cluster_assignments))
        
        for i in range(n):
            cluster = int(cluster_assignments[i])
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(texts[i])
        
        cluster_common_words = {}
        for cluster, cluster_texts in clusters.items():
            all_text = " ".join(cluster_texts)
            
            words = all_text.lower().split()
            
            stopwords = set(['dan', 'yang', 'adalah', 'dari', 'dengan', 'untuk', 'ini', 'itu', 'pada', 'the', 'a', 'an', 'is', 'of', 'in', 'to', 'and'])
            filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
            
            word_counts = Counter(filtered_words)
        
            top_words = word_counts.most_common(20)
            cluster_common_words[cluster] = top_words
        
        return clusters, cluster_common_words

    def analyze_clusters(self, model_dec=None, dataset=None, texts=None, cuda=True):
        if model_dec is None:
            model_dec = self.model_dec
        if dataset is None:
            dataset = self.dataset
        if texts is None:
            texts = self.texts
            
        cluster_assignments = self.get_cluster_assignments(model_dec, dataset, cuda=cuda)
        text_clusters, cluster_words = self.map_texts_to_clusters(texts, cluster_assignments)
    
        df_clusters = pd.DataFrame([
            {"Cluster": cluster, "Common Words": ", ".join([f"{word} ({count})" for word, count in words[:10]])}
            for cluster, words in cluster_words.items()
        ]).sort_values(by=['Cluster']).reset_index(drop=True)
        
        print(df_clusters)
        
        return df_clusters

# Here we'll add a convenient model factory function
def create_model(texts, labels, bert_model="indolem/indobert-base-uncased", cuda=True):
    """Factory function to create a new FNNJST model instance"""
    return FNNJST(texts, labels, bert_model, cuda)