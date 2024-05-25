from django.db import models

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "NLP-LTU/bertweet-large-sexism-detector"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

model.eval()

def detect_sexism(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get prediction
    predicted_label = torch.argmax(outputs.logits, dim=1).item()

    # (0: non-sexist, 1: sexist)
    if predicted_label == 1:
        return "The text contains sexism."
    else:
        return "The text does not contain sexism."

    
