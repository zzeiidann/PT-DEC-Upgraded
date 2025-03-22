import torch
import pickle
import os
import sys

# Import the model - Consider restructuring to avoid this sys.path.append
from .model import FNNJST, create_model

def load_model(model_path, texts=None, labels=None, bert_model="indolem/indobert-base-uncased", cuda=True):
    """
    Load a pretrained FNNJST model
    
    Args:
        model_path: Path to the saved model file
        texts: Optional list of texts to use for creating a new model
        labels: Optional list of labels to use for creating a new model
        bert_model: BERT model name to use
        cuda: Whether to use GPU acceleration
        
    Returns:
        model: Loaded FNNJST model
    """
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        if texts is None or labels is None:
            raise ValueError("Model file not found. To create a new model, texts and labels must be provided")
        print("Creating a new model instead...")
        return create_model(texts, labels, bert_model, cuda)
    
    try:
        # Try to load with torch first
        model_data = torch.load(model_path, map_location=torch.device('cpu'))
        print("Successfully loaded model with torch.load")
    except Exception as e:
        print(f"Error loading with torch.load: {e}")
        try:
            # Try pickle as a fallback
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            print("Successfully loaded model with pickle.load")
        except Exception as e2:
            print(f"Error loading with pickle.load: {e2}")
            print("This could be due to model file corruption, incompatible pickle protocol, "
                  "or missing dependencies required by the saved model.")
            model_data = None
    
    if model_data is None:
        print("Failed to load model. Creating a new one.")
        if texts is None or labels is None:
            raise ValueError("To create a new model, texts and labels must be provided")
        return create_model(texts, labels, bert_model, cuda)
    
    # If model_data is the entire FNNJST instance
    if isinstance(model_data, FNNJST):
        model = model_data
        # Update device and CUDA settings
        model.cuda = cuda
        model.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        # Move models to the appropriate device
        if model.model_sentiment is not None:
            model.model_sentiment = model.model_sentiment.to(model.device)
        if model.model_dec is not None:
            model.model_dec = model.model_dec.to(model.device)
        return model

    
    if 'config' in model_data:
        config = model_data['config']
        model = FNNJST(
            bert_model=config.get('bert_model', bert_model),
            num_labels=config.get('num_labels', 2),
            cuda=cuda
        )
    else:
        if texts is None or labels is None:
            raise ValueError("To create a new model without config, texts and labels must be provided")
        model = create_model(texts, labels, bert_model, cuda)
    
    if 'sentiment_model_state' in model_data:
        model.model_sentiment.load_state_dict(model_data['sentiment_model_state'])
    if 'dec_model_state' in model_data:
        model.model_dec.load_state_dict(model_data['dec_model_state'])
    
    # Move models to the appropriate device
    model.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    if model.model_sentiment is not None:
        model.model_sentiment = model.model_sentiment.to(model.device)
    if model.model_dec is not None:
        model.model_dec = model.model_dec.to(model.device)
    
    if 'tokenizer' in model_data:
        model.tokenizer = model_data['tokenizer']
    if 'label_map' in model_data:
        model.label_map = model_data['label_map']
    
    model.cuda = cuda
    
    if model.model_sentiment is not None:
        model.model_sentiment.eval()
    if model.model_dec is not None:
        model.model_dec.eval()

    return model