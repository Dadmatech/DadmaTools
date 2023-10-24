"""
Entry point for training and evaluating a lemmatizer.

This lemmatizer combines a neural sequence-to-sequence architecture with an `edit` classifier 
and two dictionaries to produce robust lemmas from word forms.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

import logging
import os
# import shutil
# import time 
# from datetime import datetime
import argparse
import numpy as np
import random
import torch
# from torch import nn, optim

from models.lemma.data import DataLoader
# from models.lemma.vocab import Vocab
from models.lemma.trainer import Trainer
from models.lemma import edit
# from models.lemma import scorer
# from models.common import utils
# import models.common.seq2seq_constant as constant
from models.common.doc import *
# from utils.conll import CoNLL
# from models import _training_logging

from dadmatools.models.common.doc import Document

import dadmatools.models.download as dl

logger = logging.getLogger('stanza')

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/lemma', help='Directory for all lemma data.')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--gold_file', type=str, default=None, help='Output CoNLL-U file.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', default='fa_ewt', type=str, help='Language')

    parser.add_argument('--no_dict', dest='ensemble_dict', action='store_false', help='Do not ensemble dictionary with seq2seq. By default use ensemble.')
    parser.add_argument('--dict_only', default='False', action='store_true', help='Only train a dictionary-based lemmatizer.')

    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--emb_dim', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--emb_dropout', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max_dec_len', type=int, default=50)
    parser.add_argument('--beam_size', type=int, default=1)

    parser.add_argument('--attn_type', default='soft', choices=['soft', 'mlp', 'linear', 'deep'], help='Attention type')
    parser.add_argument('--pos_dim', type=int, default=50)
    parser.add_argument('--pos_dropout', type=float, default=0.5)
    parser.add_argument('--no_edit', dest='edit', action='store_false', help='Do not use edit classifier in lemmatization. By default use an edit classifier.')
    parser.add_argument('--num_edit', type=int, default=len(edit.EDIT_TO_ID))
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--no_pos', dest='pos', action='store_false', help='Do not use UPOS in lemmatization. By default UPOS is used.')
    parser.add_argument('--no_copy', dest='copy', action='store_false', help='Do not use copy mechanism in lemmatization. By default copy mechanism is used to improve generalization.')

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--decay_epoch', type=int, default=30, help="Decay the lr starting from this epoch.")
    parser.add_argument('--num_epoch', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models/fa_lemmatizer/fa_lemmatizer.pt', help='Root dir for saving models.')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

    args = parser.parse_args()
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
        
    # file paths
    model_file = os.path.join(args.save_dir, '{}_lemmatizer.pt'.format(args.lang))

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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)
        
    # file paths
#     model_file = os.path.join(args.save_dir, '{}_lemmatizer.pt'.format(args.lang))

    # load model
    use_cuda = args.cuda and not args.cpu
    trainer = Trainer(model_file=args.save_dir, use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab
    
    return (trainer, args)

def lemma(trainer, args, input_tokens):
    
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


if __name__ == '__main__':
    preds = lemmatize([['رفته', 'کرده', 'خواندم']])
#     preds = lemmatize([['من کتاب را خواندم']])
    print(preds)
