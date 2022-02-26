import NERDA
import torch
import glob
from pathlib import Path
import dadmatools.pipeline.download as dl
import dadmatools.models.tokenizer as tokenizer

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    mode = 'gpu'
else:
    device = torch.device('cpu')
    mode = 'cpu'

def get_config():
  config = {
        "save_model": "saved_models/kasreh_ezafeh/kasreh_ezafeh.pt",
  }
  return config

def load_model():
    ## donwload the model (if it is not exist it'll download otherwise it dose not)
    dl.download_model('kasreh_ezafeh')
    
    args = get_config()

    prefix = str(Path(__file__).parent.absolute()).replace('models', '')
    args['save_model'] = prefix + args['save_model']

    model = torch.load(args['save_model'], map_location=device)
    model.network.device = device

    return model


def kasreh_ezafe(model, text):

    word_tokenize = lambda x: x
    sent_tokenize = tokenizer.tokenize

    resp = model.predict_text(text, sent_tokenize=sent_tokenize, word_tokenize=word_tokenize)

    sentences = resp[0]
    tags = resp[1]
    sent_tags = zip(sentences, tags)

    all_sent_tags = []
    for sent, tags in sent_tags:
        tmp = []
        for token, t in zip(sent,tags):
            tmp.append((token, t))
        all_sent_tags.append(tmp)

    return all_sent_tags
