"""
Entry point for training and evaluating a lemmatizer.

This lemmatizer combines a neural sequence-to-sequence architecture with an `edit` classifier 
and two dictionaries to produce robust lemmas from word forms.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

import logging
import sys
import os
# import shutil
# import time 
# from datetime import datetime
import argparse
import numpy as np
import random
import torch
from pathlib import Path
# from torch import nn, optim

from dadmatools.models.lemma.data import DataLoader
# from models.lemma.vocab import Vocab
from dadmatools.models.lemma.trainer import Trainer
from dadmatools.models.lemma import edit
# from models.lemma import scorer
# from models.common import utils
# import models.common.seq2seq_constant as constant
from dadmatools.models.common.doc import *
# from utils.conll import CoNLL
# from models import _training_logging

from dadmatools.models.common.doc import Document

import dadmatools.pipeline.download as dl

logger = logging.getLogger('stanza')

def parse_args():
    args = {
        'data_dir':None,
        'train_file':None,
        'eval_file':None,
        'output_file':None,
        'gold_file':None,
        'mode':'predict',
        'lang':'fa',
        'no_dict':'ensemble_dict',
        'dict_only':False,
        'hidden_dim':200,
        'emb_dim':50,
        'num_layers':1.0,
        'emb_dropout':0.5,
        'dropout':0.5,
        'max_dec_len':50,
        'beam_size':1,
        'attn_type':'spft',
        'pos_dim':50,
        'pos_dropout':0.5,
        'no_edit':False,
        'num_edit':len(edit.EDIT_TO_ID),
        'alpha':1.0,
        'no_pos':'pos',
        'no_copy':'copy',
        'sample_train':1.0, 
        'optim':'adam',
        'lr':1e-3,
        'lr_decay':0.9, 
        'decay_epoch':30, 
        'num_epoch':60,
        'batch_size':50, 
        'max_grad_norm':0.5,
        'log_step':20,
        'seed':1234,
        'cuda':torch.cuda.is_available(),
        'cpu':True,
        'save_dir':'saved_models/fa_lemmatizer/fa_lemmatizer.pt'
    }
    return args
        
def lemmatize(input_tokens):
    '''input_tokens stores list of all tokens in the sentences e.g. input_tokens = [['this', 'is', 'a', 'test', '.']] '''
    
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    prefix = str(Path(__file__).parent.absolute()).replace('models', '')
    args['save_dir'] = prefix + args['save_dir']
    # file paths
    model_file = os.path.join(args['save_dir'], '{}_lemmatizer.pt'.format(args.lang))

    # load model
    use_cuda = args.cuda and not args.cpu
    trainer = Trainer(model_file=model_file, use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab
    
    # load data
    input_dict = [[{"text": t} for t in l] for l in input_tokens]
    # doc = CoNLL.rawText2doc(input_dict)
    doc = Document(input_dict, text=None, comments=None)
    batch = DataLoader(doc, args.batch_size, loaded_args, vocab=vocab, evaluation=True)
    
    # skip eval if dev data does not exist
    if len(batch) == 0:
        logger.warning("there are no inputs")
        return
    
    dict_preds = trainer.predict_dict(batch.doc.get([TEXT, UPOS]))
    
    if loaded_args.get('dict_only', False):
        preds = dict_preds
    else:
#         logger.info("Running the seq2seq model...")
        preds = []
        edits = []
        for i, b in enumerate(batch):
            ps, es = trainer.predict(b, args.beam_size)
            preds += ps
            if es is not None:
                edits += es
        preds = trainer.postprocess(batch.doc.get([TEXT]), preds, edits=edits)
        
        if loaded_args.get('ensemble_dict', False):
            logger.info("[Ensembling dict with seq2seq lemmatizer...]")
            preds = trainer.ensemble(batch.doc.get([TEXT, UPOS]), preds)
    
    return preds


#########################################################################################################
###################################breaking the model into load_model and predict########################
#########################################################################################################


def load_model():
    ## donwload the model (if it is not exist it'll download otherwise it dose not)
    dl.download_model('fa_lemmatizer')
    
    args = parse_args()

    prefix = str(Path(__file__).parent.absolute()).replace('models', '')
    args['save_dir'] = prefix + args['save_dir']

    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    if args['cpu']:
        args['cuda'] = False
    elif args['cuda']:
        torch.cuda.manual_seed(args['seed'])
        
    # file paths
#     model_file = os.path.join(args.save_dir, '{}_lemmatizer.pt'.format(args.lang))

    # load model
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(model_file=args['save_dir'], use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab
    
    return (trainer, args)

def lemma(trainer, args, input_tokens):
    
    loaded_args, vocab = trainer.args, trainer.vocab
    # load data
    input_dict = [[{"text": t} for t in l] for l in input_tokens]
    # doc = CoNLL.rawText2doc(input_dict)
    doc = Document(input_dict, text=None, comments=None)
    batch = DataLoader(doc, args['batch_size'], loaded_args, vocab=vocab, evaluation=True)
    
    # skip eval if dev data does not exist
    if len(batch) == 0:
        logger.warning("there are no inputs")
        return
    
    dict_preds = trainer.predict_dict(batch.doc.get([TEXT, UPOS]))
    
    if loaded_args.get('dict_only', False):
        preds = dict_preds
    else:
#         logger.info("Running the seq2seq model...")
        preds = []
        edits = []
        for i, b in enumerate(batch):
            ps, es = trainer.predict(b, args['beam_size'])
            preds += ps
            if es is not None:
                edits += es
        preds = trainer.postprocess(batch.doc.get([TEXT]), preds, edits=edits)
        
        if loaded_args.get('ensemble_dict', False):
            logger.info("[Ensembling dict with seq2seq lemmatizer...]")
            preds = trainer.ensemble(batch.doc.get([TEXT, UPOS]), preds)
    
    return preds

