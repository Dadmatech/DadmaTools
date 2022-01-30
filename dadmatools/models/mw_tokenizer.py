"""
Entry point for training and evaluating a multi-word token (MWT) expander.

This MWT expander combines a neural sequence-to-sequence architecture with a dictionary
to decode the token into multiple words.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

import logging
import torch
import numpy as np
import random
import os
from pathlib import Path

from dadmatools.models.mwt.data import DataLoader
from dadmatools.models.mwt.vocab import Vocab
from dadmatools.models.mwt.trainer import Trainer
from dadmatools.models.mwt import scorer
from dadmatools.models.common import utils
import dadmatools.models.common.seq2seq_constant as constant
from dadmatools.models.common.doc import Document
from dadmatools.utils.conll import CoNLL

# logger = logging.getLogger('stanza')
import dadmatools.pipeline.download as dl

def parse_args():
    args = {
        'mode':'predict',
        'lang':'fa',
        'hidden_dim':100,
        'emb_dim':50,
        'num_layers':1,
        'emb_dropout':0.5,
        'dropout':0.5,
        'max_dec_len':'50',
        'beam_size':1,
        'attn_type':'soft',
        'sample_train':1.0,
        'optim':'adam', 
        'lr':1e-3,
        'lr_decay':0.9,
        'decay_epoch':30,
        'num_epoch':30,
        'batch_size':50,
        'max_grad_norm':5.0,
        'log_step':20,
        'seed':1234,
        'cuda':torch.cuda.is_available(),
        'cpu':True,
        'save_dir':'saved_models/fa_mwt/fa_mwt.pt'
    }
    return args
     
def load_model():
    ## donwload the model (if it is not exist it'll download otherwise it dose not)
    dl.download_model('fa_mwt')
    
    args = parse_args()

    # file paths
    prefix = str(Path(__file__).parent.absolute()).replace('models', '')
    args['save_dir'] = prefix + args['save_dir']
    
    # load model
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(model_file=args['save_dir'], use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab

    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand']:
            loaded_args[k] = args[k]
    
    return (trainer, args)

def mwt(trainer, args, input_tokens):
    loaded_args, vocab = trainer.args, trainer.vocab
    
    new_preds = []
    for l in input_tokens:
        tmp = []
        for t in l:
            if t[1] != 'MWT=No':
                doc = Document([[{"text": t[0], "misc": t[1]}]], text=None, comments=None)
                batch = DataLoader(doc, args['batch_size'], loaded_args, vocab=vocab, evaluation=True)
                preds = []
                for i, b in enumerate(batch):
                    preds += trainer.predict(b)
                if loaded_args.get('ensemble_dict', False):
                    preds = trainer.ensemble(batch.doc.get_mwt_expansions(evaluation=True), preds)
                tmp.extend(preds[0].split(' '))
            else:
                tmp.append(t[0])
        new_preds.append(tmp)
    
    return new_preds
