"""
Entry point for training and evaluating a neural tokenizer.

This tokenizer treats tokenization and sentence segmentation as a tagging problem, and uses a combination of
recurrent and convolutional architectures.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

import argparse
from copy import copy
import logging
import random
import numpy as np
import os
import torch
from pathlib import Path

from dadmatools.models.common import utils
from dadmatools.models.tokenization.trainer import Trainer
from dadmatools.models.tokenization.data import DataLoader
from dadmatools.models.tokenization.utils import load_mwt_dict, eval_model, output_predictions
# from models import _training_logging
import dadmatools.pipeline.download as dl

logger = logging.getLogger('stanza')

def parse_args():
    args = {
        'txt_file':None,
        'label_file':None,
        'mwt_json_file':None,
        'conll_file':None,
        'dev_txt_file':None,
        'dev_label_file':None,
        'dev_conll_gold':None,
        'lang':'fa',
        'shorthand':'',
        'mode':'predict',
        'skip_newline':'store_true',
        'emb_dim':32,
        'hidden_dim':64,
        'conv_filters':"1,9",
        'no-residual':'residual',
        'no-hierarchical':'hierarchical',
        'hier_invtemp':0.5,
        'input_dropout':'store_true',
        'conv_res':None,
        'rnn_layers':1,
        'max_grad_norm':1.0,
        'anneal':0.999,
        'anneal_after':2000,
        'lr0':2e-3,
        'dropout':0.33,
        'unit_dropout':0.33,
        'tok_noise':0.02, 
        'sent_drop_prob':0.2,
        'weight_decay':0.0,
        'max_seqlen':100, 
        'batch_size':32, 
        'epochs':10,
        'steps':50000, 
        'report_steps':20,
        'shuffle_steps':100,
        'eval_steps':200,
        'max_steps_before_stop':5000,
        'save_name':'None', 
        'load_name':'None',
        'save_dir':'saved_models/fa_tokenizer/fa_tokenizer.pt',
        'cuda':torch.cuda.is_available(),
        'cpu':True,
        'seed':1234,
        'use_mwt':True,
        'no_use_mwt':'use_mwt'
    }
    return args
    
def tokenize(input_sentence):
    args = parse_args()

    if args['cpu']:
        args['cuda'] = False
    utils.set_random_seed(args['seed'], args['cuda'])

    
    mwt_dict = load_mwt_dict(args['mwt_json_file'])
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(model_file=args['save_dir'], use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab
    
    for k in loaded_args:
        if not k.endswith('_file') and k not in ['cuda', 'mode', 'save_dir', 'load_name', 'save_name']:
            args[k] = loaded_args[k]
    
    batches = DataLoader(args, input_text=input_sentence, vocab=vocab, evaluation=True)
    preds = output_predictions(args['conll_file'], trainer, batches, vocab, mwt_dict, args['max_seqlen'])
    preds = [[p['text'] for p in pred] for pred in preds]
    
    return preds

#########################################################################################################
###################################breaking the model into load_model and predict########################
#########################################################################################################


def load_model():
    ## donwload the model (if it is not exist it'll download otherwise it dose not)
    dl.download_model('fa_tokenizer')
    
    args = parse_args()
    
    prefix = str(Path(__file__).parent.absolute()).replace('models', '')
    args['save_dir'] = prefix + args['save_dir']

    if args['cpu']:
        args['cuda'] = False
    utils.set_random_seed(args['seed'], args['cuda'])

    
    mwt_dict = load_mwt_dict(args['mwt_json_file'])
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(model_file=args['save_dir'], use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab
    
    for k in loaded_args:
        if not k.endswith('_file') and k not in ['cuda', 'mode', 'save_dir', 'load_name', 'save_name']:
            args[k] = loaded_args[k]
    
    return (trainer, args)

def tokenizer(trainer, args, input_sentence):
    mwt_dict = load_mwt_dict(args['mwt_json_file'])
    use_cuda = args['cuda'] and not args['cpu']
    loaded_args, vocab = trainer.args, trainer.vocab
    
    for k in loaded_args:
        if not k.endswith('_file') and k not in ['cuda', 'mode', 'save_dir', 'load_name', 'save_name']:
            args[k] = loaded_args[k]
    
    batches = DataLoader(args, input_text=input_sentence, vocab=vocab, evaluation=True)
    preds = output_predictions(args['conll_file'], trainer, batches, vocab, mwt_dict, args['max_seqlen'])
    # preds = [[p['text'] for p in pred] for pred in preds]
    new_preds = []
    for pred in preds:
        ps = []
        for p in pred:
            try:
                ps.append((p['text'], p['misc']))
            except:
                ps.append((p['text'], 'MWT=No'))
        new_preds.append(ps)
    preds = new_preds
    
    return preds

