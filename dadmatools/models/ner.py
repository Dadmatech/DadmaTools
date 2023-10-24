from __future__ import unicode_literals
# from hazm import *
# normalizer = Normalizer()

from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForTokenClassification
import torch

from pathlib import Path

import dadmatools.models.download as dl

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def get_config():
    config = {
        'save_dir':'saved_models/ner/ner/'
    }
    return config

def load_model():
    dl.download_model('ner', process_func=dl._unzip_process_func)
    
    config = get_config()
    prefix = str(Path(__file__).parent.absolute()).replace('models', '')

    model_name = prefix + config['save_dir']
    
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.to(device)
    labels = list(config.label2id.keys())

    nlp = (model, tokenizer, labels)
    
    return nlp

def ner(nlp, sentence):
    # sentence = normalizer.normalize(sentence)

    model, tokenizer, labels = nlp
    
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(device)
    outputs = model(inputs)[0]
    predictions = torch.argmax(outputs, axis=2)
    predictions = [(token, labels[prediction]) for token, prediction in zip(tokens, predictions[0].cpu().numpy())]
    
    return predictions
