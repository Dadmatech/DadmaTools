from supar import Parser
import re
import string
from copy import copy
import numpy as np
import torch
from pathlib import Path
import dadmatools.pipeline.download as dl                   

  
def parse_args():
    args = {
        'save_name':'None', 
        'load_name':'crf-con-en',
        'save_dir':'saved_models/fa_constituency/fa_constituency.pt',
        'cuda':torch.cuda.is_available(),
        'cpu':True,
        'seed':1234,
    }
    return args


def load_model():
    ## donwload the model (if it is not exist it'll download otherwise it dose not)
    dl.download_model('fa_constituency')
    
    args = parse_args()
    
    model_name = args['load_name']
    parser = Parser.load(model_name)
    
    return parser



def cons_parser(model,input_sentence):
    tokens = input_sentence.split()
    pred = model.predict(tokens,verbose=False)[0]
    return pred
    

def chunker(const_output):

    const_output = str(const_output)
    const_output = re.sub(r"(?<=\()[^\(\)\s]+(?=\s[^(])", "_", const_output)
    tags = re.findall(r"(?<=\()[^\(\)\s]+(?=\s\(\_)",const_output)
    reslist = re.findall(r'\([^a-zA-Z]+\)',const_output)
    res = [i.replace('_','').replace('(','').replace(')','') for i in reslist]
    out_str = ''
    for i,k in zip(res,tags):
        tmp = ''
        for j in i:
            if j in (string.punctuation+'،؛؟'):
                tmp = '[' + i.replace(j,'').strip() + ' ' + k + '] '  + j+' '
        if tmp!='':
            out_str = out_str + tmp
        else:
            out_str = out_str + '[' + i + ' ' + k +'] '

    return out_str
    
    