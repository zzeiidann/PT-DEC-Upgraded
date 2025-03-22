from setuptools import setup, find_packages

setup(
    name="FNNJST",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fusion Neural Network for Joint Sentiment and Topic Modeling",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FNNJST",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "transformers>=4.5.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "ptdec>=0.1.0",
        "ptsdae>=0.1.0",
    ],
)