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

from models.common import utils
from models.tokenization.trainer import Trainer
from models.tokenization.data import DataLoader
from models.tokenization.utils import load_mwt_dict, eval_model, output_predictions
# from models import _training_logging
import pipeline.download as dl

logger = logging.getLogger('stanza')

def parse_args(args=None):
    """
    If args == None, the system args are used.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_file', type=str, help="Input plaintext file")
    parser.add_argument('--label_file', type=str, default=None, help="Character-level label file")
    parser.add_argument('--mwt_json_file', type=str, default=None, help="JSON file for MWT expansions")
    parser.add_argument('--conll_file', type=str, default=None, help="CoNLL file for output")
    parser.add_argument('--dev_txt_file', type=str, help="(Train only) Input plaintext file for the dev set")
    parser.add_argument('--dev_label_file', type=str, default=None, help="(Train only) Character-level label file for the dev set")
    parser.add_argument('--dev_conll_gold', type=str, default=None, help="(Train only) CoNLL-U file for the dev set for early stopping")
    parser.add_argument('--lang', default='fa', type=str, help="Language")
    parser.add_argument('--shorthand', type=str, help="UD treebank shorthand")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--skip_newline', action='store_true', help="Whether to skip newline characters in input. Particularly useful for languages like Chinese.")

    parser.add_argument('--emb_dim', type=int, default=32, help="Dimension of unit embeddings")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Dimension of hidden units")
    parser.add_argument('--conv_filters', type=str, default="1,9", help="Configuration of conv filters. ,, separates layers and , separates filter sizes in the same layer.")
    parser.add_argument('--no-residual', dest='residual', action='store_false', help="Add linear residual connections")
    parser.add_argument('--no-hierarchical', dest='hierarchical', action='store_false', help="\"Hierarchical\" RNN tokenizer")
    parser.add_argument('--hier_invtemp', type=float, default=0.5, help="Inverse temperature used in propagating tokenization predictions between RNN layers")
    parser.add_argument('--input_dropout', action='store_true', help="Dropout input embeddings as well")
    parser.add_argument('--conv_res', type=str, default=None, help="Convolutional residual layers for the RNN")
    parser.add_argument('--rnn_layers', type=int, default=1, help="Layers of RNN in the tokenizer")

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm to clip to")
    parser.add_argument('--anneal', type=float, default=.999, help="Anneal the learning rate by this amount when dev performance deteriorate")
    parser.add_argument('--anneal_after', type=int, default=2000, help="Anneal the learning rate no earlier than this step")
    parser.add_argument('--lr0', type=float, default=2e-3, help="Initial learning rate")
    parser.add_argument('--dropout', type=float, default=0.33, help="Dropout probability")
    parser.add_argument('--unit_dropout', type=float, default=0.33, help="Unit dropout probability")
    parser.add_argument('--tok_noise', type=float, default=0.02, help="Probability to induce noise to the input of the higher RNN")
    parser.add_argument('--sent_drop_prob', type=float, default=0.2, help="Probability to drop sentences at the end of batches during training uniformly at random.  Idea is to fake paragraph endings.")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--max_seqlen', type=int, default=100, help="Maximum sequence length to consider at a time")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use")
    parser.add_argument('--epochs', type=int, default=10, help="Total epochs to train the model for")
    parser.add_argument('--steps', type=int, default=50000, help="Steps to train the model for, if unspecified use epochs")
    parser.add_argument('--report_steps', type=int, default=20, help="Update step interval to report loss")
    parser.add_argument('--shuffle_steps', type=int, default=100, help="Step interval to shuffle each paragraph in the generator")
    parser.add_argument('--eval_steps', type=int, default=200, help="Step interval to evaluate the model on the dev set for early stopping")
    parser.add_argument('--max_steps_before_stop', type=int, default=5000, help='Early terminates after this many steps if the dev scores are not improving')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--load_name', type=str, default=None, help="File name to load a saved model")
    parser.add_argument('--save_dir', type=str, default='saved_models/fa_tokenizer/fa_tokenizer.pt', help="Directory to save models in")
#     parser.add_argument('--save_dir', type=str, default='saved_models/tokenize/fa_ewt_tokenizer.pt', help="Directory to save models in")
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA and run on CPU.')
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--use_mwt', dest='use_mwt', default=None, action='store_true', help='Whether or not to include mwt output layers.  If set to None, this will be determined by examining the training data for MWTs')
    parser.add_argument('--no_use_mwt', dest='use_mwt', action='store_false', help='Whether or not to include mwt output layers')

    args = parser.parse_args(args=args)
    return args
    
def tokenize(input_sentence):
    args = parse_args()

    if args.cpu:
        args.cuda = False
    utils.set_random_seed(args.seed, args.cuda)

    args = vars(args)
    
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

    if args.cpu:
        args.cuda = False
    utils.set_random_seed(args.seed, args.cuda)

    args = vars(args)
    
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
    preds = [[p['text'] for p in pred] for pred in preds]
    return preds

if __name__ == '__main__':
    preds = tokenize('دیروز با مهدی و علی به کتابخانه رفتیم. سلام ما را به او برسان')
    print(preds)
