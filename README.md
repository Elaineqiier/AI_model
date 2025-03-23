# AI_model
AI Language Model for detecting xenophobic language
Project Summary

This project implements an enhanced hate speech detection system. It uses a pre-trained BERT model (from transformers library) fine-tuned for classifying text into different categories of hate speech (toxic, severe toxic, obscene, threat, insult, identity hate).

The key features are:

Text and Document Analysis: Able to analyze individual text strings and content from various file formats (.txt, .docx, .pdf).

Severity Assessment: Assigns a severity level ("High", "Medium", "Low", "None") based on the predicted hate speech probabilities.

Toxic Category Identification: Identifies specific categories of hate speech present in the text.

Summarization: Generates a concise summary of the analysis results, including the severity level and toxic categories detected.

Pre-training and Fine-Tuning: The code utilizes the standard architecture for loading the BERT model and modifying the final layer to have six different output labels, and use training dataset to fine-tune it.

# Enhanced Hate Speech Detection System

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![transformers](https://img.shields.io/badge/transformers-4.44.2-green)](https://huggingface.co/transformers)
[![PyTorch](https://img.shields.io/badge/PyTorch->=1.0-orange.svg)](https://pytorch.org/)

## Overview

This project implements an enhanced hate speech detection system designed to identify and categorize toxic content in text and documents. The system leverages a pre-trained BERT model, fine-tuned for classifying text into categories of hate speech. It offers detailed analysis reports, including severity assessment and identification of toxic categories.

## Features

*   **Text Analysis:** Analyzes individual text strings for hate speech.
*   **Document Analysis:** Supports analysis of various file formats: `.txt`, `.docx`, and `.pdf`.
*   **Severity Assessment:** Assigns severity levels ("High", "Medium", "Low", "None") based on the predicted hate speech probabilities.
*   **Toxic Category Identification:** Pinpoints specific categories of hate speech present in the analyzed content (toxic, severe\_toxic, obscene, threat, insult, identity\_hate).
*   **Summarization:** Generates a summary of the analysis, including severity and categories.
*   **Pre-trained BERT Model:** Uses the power of transfer learning to achieve robust hate speech detection.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [repository_url]
    cd [repository_name]
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    Or, execute this code for installation

    ```python
    def setup_requirements():
        """Install required dependencies"""
        import subprocess
        import sys

        requirements = [
            'python-docx',
            'PyPDF2',
            'nltk',
            'transformers',
            'torch',
            'tqdm'
        ]
        for package in requirements:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except:
                print(f"Failed to install {package}")
    ```

3.  **Download NLTK resources (if not already present):**

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## Usage

1.  **Example Usage (from code):**

    ```python
    from google.colab import drive
    import torch
    import pandas as pd
    import numpy as np
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from tqdm.notebook import tqdm
    import os
    from google.colab import files, drive
    import json
    import re
    from typing import List, Dict, Any
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from collections import Counter
    import docx
    import PyPDF2
    import nltk
    import subprocess
    import sys
    from docx import Document
    import PyPDF2
    import os


    def setup_requirements():
        """Install required dependencies"""
        requirements = [
            'python-docx',
            'PyPDF2',
            'nltk',
            'transformers',
            'torch',
            'tqdm'
        ]
        for package in requirements:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except:
                print(f"Failed to install {package}")


    class DocumentProcessor():
        """Document processor for file handling"""
        def __init__(self):
            self.supported_formats = {
                '.txt': self._read_txt,
                '.docx': self._read_docx,
                '.pdf': self._read_pdf
            }
        def _read_txt(self, file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        def _read_docx(self, file_path):
            doc = Document(file_path)
            return '\\n'.join([paragraph.text for paragraph in doc.paragraphs])
        def _read_pdf(self, file_path):
            content = []
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    content.append(page.extract_text())
            return '\\n'.join(content)
        def read_file(self, file_path):
            """Read file content"""
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")
            return self.supported_formats[file_ext](file_path)


    class HateSpeechDetector():
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
            self.model.to(self.device)
            self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        def analyze_text(self, text):
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1).cpu().detach().numpy()[0]
            predictions = {label: probability for label, probability in zip(self.label_columns, probabilities)}
            severity = self._determine_severity(predictions)
            toxic_categories = [label for label, score in predictions.items() if score >= 0.5]
            return {
                'text': text,
                'severity': severity,
                'predictions': predictions,
                'toxic_categories': toxic_categories,
                'summary': self._generate_summary(toxic_categories, severity)
            }

        def _determine_severity(self, predictions):
            max_probability = max(predictions.values())
            if max_probability >= 0.8:
                return "High"
            elif max_probability >= 0.6:
                return "Medium"
            elif max_probability >= 0.4:
                return "Low"
            else:
                return "None"

        def _generate_summary(self, toxic_categories, severity):
            if not toxic_categories:
                return "No concerning content detected."
            summary = f"Detected {severity.lower()} level of concerning content, including: {', '.join(toxic_categories)}."
            return summary

        def load_dataset(self):
            """Load dataset if you have a separate training dataset"""
            return None, None

        def train_model(self, texts, labels, epochs=3, batch_size=16, learning_rate=2e-5):
            """Train the model using text and labels. Override in subclass for training with real data"""
            print("Implement training logic in a subclass")

        def save_model(self, model_path='/content/drive/MyDrive/hate_speech_model.pth'):
            """Save the model for later use"""
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")

        def load_pretrained_model(self, model_path):
            """Load a pretrained model"""
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()


    class EnhancedHateSpeechDetector(HateSpeechDetector):
        def __init__(self, model_path=None):
            super().__init__()
            self.document_processor = DocumentProcessor()
            self.initialize_training_data()
            if model_path:
                self.load_pretrained_model(model_path)

        def initialize_training_data(self):
            """Initialize training data"""
            self.training_data = {
                'toxic': [
                    "savage tribes", "barbaric people", "primitive cultures",
                    "uncivilized groups", "inferior races", "hostile natives"
                ],
                'severe_toxic': [
                    "complete elimination", "total destruction", "absolute conquest",
                    "racial superiority", "ethnic cleansing"
                ],
                'obscene': [
                    "degrading behavior", "immoral conduct", "corrupt practices"
                ],
                'threat': [
                    "must be eliminated", "should be destroyed", "will be conquered",
                    "need to be subdued"
                ],
                'insult': [
                    "backward people", "uncivilized beings", "primitive minds"
                ],
                'identity_hate': [
                    "racial inferiority", "ethnic hatred", "cultural supremacy"
                ]
            }

        def train_with_examples(self):
            """Train using example data"""
            texts = []
            labels = []
            for category, examples in self.training_data.items():
                for example in examples:
                    texts.append(example)
                    label = [0] * len(self.label_columns)
                    label[self.label_columns.index(category)] = 1
                    labels.append(label)
            return self.train_model(texts, labels, epochs=3)

        def analyze_file(self, file_path):
            """Analyze file content"""
            try:
                content = self.document_processor.read_file(file_path)
                paragraphs = content.split('\\n\\n')
                results = []
                for paragraph in paragraphs:
                    if paragraph.strip():
                        result = self.analyze_text(paragraph)
                        if result['severity'] != 'None':
                            results.append(result)
                return self._summarize_results(results, content)
            except Exception as e:
                print(f"Error analyzing file: {str(e)}")
                return None

        def _summarize_results(self, results, full_content):
            """Summarize analysis results"""
            if not results:
                return {
                    'content': full_content,
                    'severity': 'None',
                    'toxic_sections': [],
                    'overall_score': 0,
                    'summary': "No significant concerning content detected."
                }
            max_severity = max(
                (r['severity'] for r in results),
                key=lambda x: {'High': 3, 'Medium': 2, 'Low': 1, 'None': 0}.get(x, 0)
            )
            toxic_sections = [
                {
                    'text': r['text'],
                    'severity': r['severity'],
                    'categories': r['toxic_categories']
                }
                for r in results
            ]
            scores = [
                max(r['predictions'].values()) for r in results
            ]
            overall_score = (sum(scores) / len(scores)) * 10
            return {
                'content': full_content,
                'severity': max_severity,
                'toxic_sections': toxic_sections,
                'overall_score': round(overall_score, 1),
                'summary': self._generate_summary(toxic_sections, max_severity)
            }
        def save_model(model, model_path='/content/drive/MyDrive/hate_speech_model.pth'):
            """Saves the PyTorch model to the specified path."""
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")

        def load_model(model_path='/content/drive/MyDrive/hate_speech_model.pth'):
            """Loads a PyTorch model from the specified path."""
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)  # You might need to adjust this based on your model
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model

    drive.mount('/content/drive')


    # Example usage (assuming 'model' is your trained model):
    # save_model(model)

    # Later, to load the model:
    # loaded_model = load_model()
    from google.colab import files, drive
    drive.mount('/content/drive')

    def setup_requirements():
        """Install required dependencies"""
        import subprocess
        import sys

        requirements = [
            'python-docx',
            'PyPDF2',
            'nltk',
            'transformers',
            'torch',
            'tqdm'
        ]
        for package in requirements:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except:
                print(f"Failed to install {package}")


    class DocumentProcessor():
        """Document processor for file handling"""
        def __init__(self):
            self.supported_formats = {
                '.txt': self._read_txt,
                '.docx': self._read_docx,
                '.pdf': self._read_pdf
            }
        def _read_txt(self, file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        def _read_docx(self, file_path):
            doc = Document(file_path)
            return '\\n'.join([paragraph.text for paragraph in doc.paragraphs])
        def _read_pdf(self, file_path):
            content = []
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    content.append(page.extract_text())
            return '\\n'.join(content)
        def read_file(self, file_path):
            """Read file content"""
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")
            return self.supported_formats[file_ext](file_path)


    class HateSpeechDetector():
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
            self.model.to(self.device)
            self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        def analyze_text(self, text):
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1).cpu().detach().numpy()[0]
            predictions = {label: probability for label, probability in zip(self.label_columns, probabilities)}
            severity = self._determine_severity(predictions)
            toxic_categories = [label for label, score in predictions.items() if score >= 0.5]
            return {
                'text': text,
                'severity': severity,
                'predictions': predictions,
                'toxic_categories': toxic_categories,
                'summary': self._generate_summary(toxic_categories, severity)
            }

        def _determine_severity(self, predictions):
            max_probability = max(predictions.values())
            if max_probability >= 0.8:
                return "High"
            elif max_probability >= 0.6:
                return "Medium"
            elif max_probability >= 0.4:
                return "Low"
            else:
                return "None"

        def _generate_summary(self, toxic_categories, severity):
            if not toxic_categories:
                return "No concerning content detected."
            summary = f"Detected {severity.lower()} level of concerning content, including: {', '.join(toxic_categories)}."
            return summary

        def load_dataset(self):
            """Load dataset if you have a separate training dataset"""
            return None, None

        def train_model(self, texts, labels, epochs=3, batch_size=16, learning_rate=2e-5):
            """Train the model using text and labels. Override in subclass for training with real data"""
            print("Implement training logic in a subclass")

        def save_model(self, model_path='/content/drive/MyDrive/hate_speech_model.pth'):
            """Save the model for later use"""
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")

        def load_pretrained_model(self, model_path):
            """Load a pretrained model"""
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()


    class EnhancedHateSpeechDetector(HateSpeechDetector):
        def __init__(self, model_path=None):
            super().__init__()
            self.document_processor = DocumentProcessor()
            self.initialize_training_data()
            if model_path:
                self.load_pretrained_model(model_path)

        def initialize_training_data(self):
            """Initialize training data"""
            self.training_data = {
                'toxic': [
                    "savage tribes", "barbaric people", "primitive cultures",
                    "uncivilized groups", "inferior races", "hostile natives"
                ],
                'severe_toxic': [
                    "complete elimination", "total destruction", "absolute conquest",
                    "racial superiority", "ethnic cleansing"
                ],
                'obscene': [
                    "degrading behavior", "immoral conduct", "corrupt practices"
                ],
                'threat': [
                    "must be eliminated", "should be destroyed", "will be conquered",
                    "need to be subdued"
                ],
                'insult': [
                    "backward people", "uncivilized beings", "primitive minds"
                ],
                'identity_hate': [
                    "racial inferiority", "ethnic hatred", "cultural supremacy"
                ]
            }

        def train_with_examples(self):
            """Train using example data"""
            texts = []
            labels = []
            for category, examples in self.training_data.items():
                for example in examples:
                    texts.append(example)
                    label = [0] * len(self.label_columns)
                    label[self.label_columns.index(category)] = 1
                    labels.append(label)
            return self.train_model(texts, labels, epochs=3)

        def analyze_file(self, file_path):
            """Analyze file content"""
            try:
                content = self.document_processor.read_file(file_path)
                paragraphs = content.split('\\n\\n')
                results = []
                for paragraph in paragraphs:
                    if paragraph.strip():
                        result = self.analyze_text(paragraph)
                        if result['severity'] != 'None':
                            results.append(result)
                return self._summarize_results(results, content)
            except Exception as e:
                print(f"Error analyzing file: {str(e)}")
                return None

        def _summarize_results(self, results, full_content):
            """Summarize analysis results"""
            if not results:
                return {
                    'content': full_content,
                    'severity': 'None',
                    'toxic_sections': [],
                    'overall_score': 0,
                    'summary': "No significant concerning content detected."
                }
            max_severity = max(
                (r['severity'] for r in results),
                key=lambda x: {'High': 3, 'Medium': 2, 'Low': 1, 'None': 0}.get(x, 0)
            )
            toxic_sections = [
                {
                    'text': r['text'],
                    'severity': r['severity'],
                    'categories': r['toxic_categories']
                }
                for r in results
            ]
            scores = [
                max(r['predictions'].values()) for r in results
            ]
            overall_score = (sum(scores) / len(scores)) * 10
            return {
                'content': full_content,
                'severity': max_severity,
                'toxic_sections': toxic_sections,
                'overall_score': round(overall_score, 1),
                'summary': self._generate_summary(toxic_sections, max_severity)
            }
        def save_model(model, model_path='/content/drive/MyDrive/hate_speech_model.pth'):
            """Saves the PyTorch model to the specified path."""
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")

        def load_model(model_path='/content/drive/MyDrive/hate_speech_model.pth'):
            """Loads a PyTorch model from the specified path."""
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)  # You might need to adjust this based on your model
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model
    # Example:
    detector = EnhancedHateSpeechDetector()
    detector.train_with_examples()

    example_text = "This is an example of hate speech. [Insert toxic content here]."
    analysis_result = detector.analyze_text(example_text)
    print(analysis_result)

    ```

2.  **Analyze text from a file:**

```python
    from google.colab import files, drive
    import torch
    import pandas as pd
    import numpy as np
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from tqdm.notebook import tqdm
    import os
    from google.colab import files, drive
    import json
    import re
    from typing import List, Dict, Any
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from collections import Counter
    import docx
    import PyPDF2
    import nltk
    import subprocess
    import sys
    from docx import Document
    import PyPDF2
    import os


    def setup_requirements():
        """Install required dependencies"""
        requirements = [
            'python-docx',
            'PyPDF2',
            'nltk',
            'transformers',
            'torch',
            'tqdm'
        ]
        for package in requirements:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except:
                print(f"Failed to install {package}")


    class DocumentProcessor():
        """Document processor for file handling"""
        def __init__(self):
            self.supported_formats = {
                '.txt': self._read_txt,
                '.docx': self._read_docx,
                '.pdf': self._read_pdf
            }
        def _read_txt(self, file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        def _read_docx(self, file_path):
            doc = Document(file_path)
            return '\\n'.join([paragraph.text for paragraph in doc.paragraphs])
        def _read_pdf(self, file_path):
            content = []
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    content.append(page.extract_text())
            return '\\n'.join(content)
        def read_file(self, file_path):
            """Read file content"""
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")
            return self.supported_formats[file_ext](file_path)


    class HateSpeechDetector():
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
            self.model.to(self.device)
            self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        def analyze_text(self, text):
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1).cpu().detach().numpy()[0]
            predictions = {label: probability for label, probability in zip(self.label_columns, probabilities)}
            severity = self._determine_severity(predictions)
            toxic_categories = [label for label, score in predictions.items() if score >= 0.5]
            return {
                'text': text,
                'severity': severity,
                'predictions': predictions,
                'toxic_categories': toxic_categories,
                'summary': self._generate_summary(toxic_categories, severity)
            }

        def _determine_severity(self, predictions):
            max_probability = max(predictions.values())
            if max_probability >= 0.8:
                return "High"
            elif max_probability >= 0.6:
                return "Medium"
            elif max_probability >= 0.4:
                return "Low"
            else:
                return "None"

        def _generate_summary(self, toxic_categories, severity):
            if not toxic_categories:
                return "No concerning content detected."
            summary = f"Detected {severity.lower()} level of concerning content, including: {', '.join(toxic_categories)}."
            return summary

        def load_dataset(self):
            """Load dataset if you have a separate training dataset"""
            return None, None

        def train_model(self, texts, labels, epochs=3, batch_size=16, learning_rate=2e-5):
            """Train the model using text and labels. Override in subclass for training with real data"""
            print("Implement training logic in a subclass")

        def save_model(self, model_path='/content/drive/MyDrive/hate_speech_model.pth'):
            """Save the model for later use"""
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")

        def load_pretrained_model(self, model_path):
            """Load a pretrained model"""
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()


    class EnhancedHateSpeechDetector(HateSpeechDetector):
        def __init__(self, model_path=None):
            super().__init__()
            self.document_processor = DocumentProcessor()
            self.initialize_training_data()
            if model_path:
                self.load_pretrained_model(model_path)

        def initialize_training_data(self):
            """Initialize training data"""
            self.training_data = {
                'toxic': [
                    "savage tribes", "barbaric people", "primitive cultures",
                    "uncivilized groups", "inferior races", "hostile natives"
                ],
                'severe_toxic': [
                    "complete elimination", "total destruction", "absolute conquest",
                    "racial superiority", "ethnic cleansing"
                ],
                'obscene': [
                    "degrading behavior", "immoral conduct", "corrupt practices"
                ],
                'threat': [
                    "must be eliminated", "should be destroyed", "will be conquered",
                    "need to be subdued"
                ],
                'insult': [
                    "backward people", "uncivilized beings", "primitive minds"
                ],
                'identity_hate': [
                    "racial inferiority", "ethnic hatred", "cultural supremacy"
                ]
            }

        def train_with_examples(self):
            """Train using example data"""
            texts = []
            labels = []
            for category, examples in self.training_data.items():
                for example in examples:
                    texts.append(example)
                    label = [0] * len(self.label_columns)
                    label[self.label_columns.index(category)] = 1
                    labels.append(label)
            return self.train_model(texts, labels, epochs=3)

        def analyze_file(self, file_path):
            """Analyze file content"""
            try:
                content = self.document_processor.read_file(file_path)
                paragraphs = content.split('\\n\\n')
                results = []
                for paragraph in paragraphs:
                    if paragraph.strip():
                        result = self.analyze_text(paragraph)
                        if result['severity'] != 'None':
                            results.append(result)
                return self._summarize_results(results, content)
            except Exception as e:
                print(f"Error analyzing file: {str(e)}")
                return None

        def _summarize_results(self, results, full_content):
            """Summarize analysis results"""
            if not results:
                return {
                    'content': full_content,
                    'severity': 'None',
                    'toxic_sections': [],
                    'overall_score': 0,
                    'summary': "No significant concerning content detected."
                }
            max_severity = max(
                (r['severity'] for r in results),
                key=lambda x: {'High': 3, 'Medium': 2, 'Low': 1, 'None': 0}.get(x, 0)
            )
            toxic_sections = [
                {
                    'text': r['text'],
                    'severity': r['severity'],
                    'categories': r['toxic_categories']
                }
                for r in results
            ]
            scores = [
                max(r['predictions'].values()) for r in results
            ]
            overall_score = (sum(scores) / len(scores)) * 10
            return {
                'content': full_content,
                'severity': max_severity,
                'toxic_sections': toxic_sections,
                'overall_score': round(overall_score, 1),
                'summary': self._generate_summary(toxic_sections, max_severity)
            }
        def save_model(model, model_path='/content/drive/MyDrive/hate_speech_model.pth'):
            """Saves the PyTorch model to the specified path."""
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")

        def load_model(model_path='/content/drive/MyDrive/hate_speech_model.pth'):
            """Loads a PyTorch model from the specified path."""
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)  # You might need to adjust this based on your model
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model

    # Load a pretrained model from Google Drive if it's available, otherwise initialize a new model
    model_path = '/content/drive/MyDrive/hate_speech_model.pth'
    detector = EnhancedHateSpeechDetector(model_path=model_path)

    uploaded = files.upload()
    for filename in uploaded.keys():
        print(f"Analyzing file: {filename}")
        analysis_result = detector.analyze_file(filename)
        print(analysis_result)

    ```
*   **Mount Google Drive to Load the Model (Colab):**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    **Load your pretrained model**:
    ```python
    from google.colab import files, drive
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

    def load_model(model, model_path='/content/drive/MyDrive/hate_speech_model.pth'):
        """Loads a PyTorch model from the specified path."""
        model.
