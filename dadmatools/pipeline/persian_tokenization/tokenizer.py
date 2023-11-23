"""
Entry point for training and evaluating a neural tokenizer.

This tokenizer treats tokenization and sentence segmentation as a tagging problem, and uses a combination of
recurrent and convolutional architectures.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

import logging
import torch
from pathlib import Path


from .trainer import Trainer
from .data import DataLoader
from .utils import load_mwt_dict, output_predictions, set_random_seed
from .download import download_model

logger = logging.getLogger('stanza')


def parse_args(cache_dir):
    args = {
        'txt_file': None,
        'label_file': None,
        'mwt_json_file': None,
        'conll_file': None,
        'dev_txt_file': None,
        'dev_label_file': None,
        'dev_conll_gold': None,
        'lang': 'fa',
        'shorthand': '',
        'mode': 'predict',
        'skip_newline': 'store_true',
        'emb_dim': 32,
        'hidden_dim': 64,
        'conv_filters': "1,9",
        'no-residual': 'residual',
        'no-hierarchical': 'hierarchical',
        'hier_invtemp': 0.5,
        'input_dropout': 'store_true',
        'conv_res': None,
        'rnn_layers': 1,
        'max_grad_norm': 1.0,
        'anneal': 0.999,
        'anneal_after': 2000,
        'lr0': 2e-3,
        'dropout': 0.33,
        'unit_dropout': 0.33,
        'tok_noise': 0.02,
        'sent_drop_prob': 0.2,
        'weight_decay': 0.0,
        'max_seqlen': 100,
        'batch_size': 32,
        'epochs': 10,
        'steps': 50000,
        'report_steps': 20,
        'shuffle_steps': 100,
        'eval_steps': 200,
        'max_steps_before_stop': 5000,
        'save_name': 'None',
        'load_name': 'None',
        'save_dir': f'{cache_dir}/fa_tokenizer.pt',
        'cuda': torch.cuda.is_available(),
        'cpu': True,
        'seed': 1234,
        'use_mwt': True,
        'no_use_mwt': 'use_mwt'
    }
    return args


def load_tokenizer_model(cache_dir: str):
    ## donwload the model (if it is not exist it'll download otherwise it dose not)
    download_model('fa_tokenizer', cache_dir=cache_dir)

    args = parse_args(cache_dir)

    if args['cpu']:
        args['cuda'] = False
    set_random_seed(args['seed'], args['cuda'])

    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(model_file=args['save_dir'], use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab

    for k in loaded_args:
        if not k.endswith('_file') and k not in ['cuda', 'mode', 'save_dir', 'load_name', 'save_name']:
            args[k] = loaded_args[k]

    return (trainer, args)


def tokenizer(trainer, args, input_sentence):
    mwt_dict = load_mwt_dict(args['mwt_json_file'])
    loaded_args, vocab = trainer.args, trainer.vocab

    for k in loaded_args:
        if not k.endswith('_file') and k not in ['cuda', 'mode', 'save_dir', 'load_name', 'save_name']:
            args[k] = loaded_args[k]

    batches = DataLoader(args, input_text=input_sentence, vocab=vocab, evaluation=True)
    preds = output_predictions(args['conll_file'], trainer, batches, vocab, mwt_dict, args['max_seqlen'])

    return [[word['text'] for word in sent] for sent in preds]
