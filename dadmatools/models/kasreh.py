import torch
import glob
from pathlib import Path
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
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
        "save_model": "saved_models/kasreh_ezafeh/kasreh_ezafeh_checkpoint",
  }
  return config

def load_model():
    ## donwload the model (if it is not exist it'll download otherwise it dose not)
    dl.download_model('kasreh_ezafeh', process_func=dl._unzip_process_func)
    
    args = get_config()

    prefix = str(Path(__file__).parent.absolute()).replace('models', '')
    args['save_model'] = prefix + args['save_model']


    model = AutoModelForTokenClassification.from_pretrained(args['save_model'])
    model.to(device)
    tokenizer =  AutoTokenizer.from_pretrained(args['save_model'], model_max_length=512)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    return ner_pipeline


def kasreh_ezafe(ner_pipeline, text):

    ent2tag = {'LABEL_0': 'KASREH', 'LABEL_1':'N-KASREH'}
    ner_resp = ner_pipeline(text)
    output = [{'tag':ent2tag[item['entity']], 'word':item['word']} for item in ner_resp]
    return output