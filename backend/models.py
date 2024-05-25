from django.db import models

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "NLP-LTU/bertweet-large-sexism-detector"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

def detect_sexism(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass through model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted label
    predicted_label = torch.argmax(outputs.logits, dim=1).item()

    # Return label (0: non-sexist, 1: sexist)
    
    if predicted_label == 1:
        return "The text contains sexism."
    else:
        return "The text does not contain sexism."

    
