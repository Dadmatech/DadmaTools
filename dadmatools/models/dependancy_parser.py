from dadmatools.models.flair import embeddings as Embeddings
import copy
from dadmatools.models.flair.data import Dictionary
from dadmatools.models.flair import models as models
from pathlib import Path
# import models.flair as flair
from dadmatools.models.flair.list_data import ListCorpus
import torch
from pathlib import Path

import dadmatools.pipeline.download as dl

def get_config():
  config = {
    "ModelFinetuner": {
      "direct_upsample_rate": -1,
      "distill_mode": False,
      "down_sample_amount": -1,
      "ensemble_distill_mode": False,
      "language_resample": False
    },
    "dependency": {
      "Corpus": "UniversalDependenciesCorpus-1",
      "UniversalDependenciesCorpus-1": {
        "data_folder": "flair/datasets/persian_dp",
        "add_root": True
      }
    },
    "embeddings-saved-dir": "saved_models/parsbert/parsbert/",
    "embeddings": {
      "BertEmbeddings-0": {
  #       "bert_model_or_path": "models/parsbert/",
        "bert_model_or_path": "saved_models/parsbert/parsbert/",
        "fine_tune": True,
        "layers": "-1",
        "pooling_operation": "mean"
      }
    },
    "model": {
      "SemanticDependencyParser": {
        "binary": False,
        "dropout": 0.0,
        "factorize": True,
        "hidden_size": 400,
        "init_std": 0.25,
        "interpolation": 0.5,
        "iterations": 3,
        "locked_dropout": 0.0,
        "lstm_dropout": 0.33,
        "mlp_dropout": 0.33,
        "n_mlp_arc": 500,
        "n_mlp_rel": 100,
        "n_mlp_sec": 150,
        "rnn_layers": 3,
        "tree": False,
        "use_cop": True,
        "use_crf": False,
        "use_gp": True,
        "use_rnn": False,
        "use_second_order": False,
        "use_sib": True,
        "word_dropout": 0.1
      }
    },
  #   "model_name": "en-bert_10epoch_0.5inter_2000batch_0.00005lr_20lrrate_ptb_monolingual_nocrf_fast_warmup_freezing_beta_weightdecay_finetune_saving_nodev_dependency16",
  #   "srl": {
  #     "Corpus": "SRL-EN"
  #   },
  #   "target_dir": "saved_models/",
    "model_name": "dependencyparser.pt",
    "srl": {
      "Corpus": "SRL-EN"
    },
    "target_dir": "saved_models/dependencyparser/",
    "targets": "dependency"
  }
  return config


def make_tag_dictionary() -> Dictionary:
        # Make the tag dictionary
        tag_dictionary: Dictionary = Dictionary()
        tag_dictionary.add_item("O")
        # list = ["<unk>", "root", "amod", "nsubj:pass", "case", "fixed", "nmod", "aux:pass", "punct", "parataxis", "nsubj", "mark", "ccomp", "cop", "obj", "aux", "advmod", "nummod", "cc", "det", "appos", "conj", "obl", "acl", "xcomp", "compound:lvc", "advcl", "flat", "dep", "vocative", "compound", "dislocated", "flat:foreign", "goeswith", "iobj"]
        list = ['goeswith', 'obl', 'cop', 'conj', 'nummod', 'flat:num', 'dep', 'obl:arg', 'compound', 'compound:lvc', 'nmod', 'advcl', 'iobj', 'flat:name', '<unk>', 'aux', 'root', 'fixed', 'xcomp', 'appos', 'csubj', 'advmod', 'nsubj:pass', 'vocative', 'punct', 'case', 'aux:pass', 'det', 'acl', 'nsubj', 'parataxis', 'cc', 'amod', 'obj', 'mark', 'ccomp']
        for t in list:
            tag_dictionary.add_item(t)
        tag_dictionary.add_item("<START>")
        tag_dictionary.add_item("<STOP>")
        # import pdb;pdb.set_trace()
        return tag_dictionary

def create_model(config):
		embeddings = config['embeddings']
		embedding_list: List[TokenEmbeddings]=[]
		for embedding in config['embeddings']:
			embedding_list.append(getattr(Embeddings,embedding.split('-')[0])(**embeddings[embedding]))
		embeddings: Embeddings.StackedEmbeddings = Embeddings.StackedEmbeddings(embeddings=embedding_list)
		kwargs=copy.deepcopy(config['model'])
		classname=list(kwargs.keys())[0]
		kwargs=copy.deepcopy(config['model'][classname])
		kwargs['embeddings']=embeddings
		kwargs['tag_type']='dependency'
		kwargs['tag_dictionary']=make_tag_dictionary()
		tagger = getattr(models,classname)(**kwargs, config=config)
		tagger.word_map = None
		tagger.char_map = None
		tagger.lemma_map = None
		tagger.postag_map = None
        
		base_path=Path(config['target_dir'])/config['model_name']

		if (base_path / "best-model.pt").exists():
# 			print('Loading pretraining best model')
			tagger = tagger.load(base_path / "best-model.pt")
		elif (base_path / "final-model.pt").exists():
# 			print('Loading pretraining final model')
			tagger = tagger.load(base_path / "final-model.pt")
		elif (base_path).exists():
			tagger = tagger.load(base_path)
		else:
			assert 0, str(base_path)+ ' not exist!'
		tagger.use_bert=False
		for embedding in config['embeddings']:
			if 'bert' in embedding.lower():
				tagger.use_bert=True
		return tagger

def dependancy_parser_model(tokens_list):
    student=create_model(config)
    # base_path=Path(config['target_dir'])/config['model_name']
    preds_arcs, preds_rels = student.predict(tokens_list,prediction_mode=True)
    preds_arcs = [p.item() for p in preds_arcs]
    return preds_arcs, preds_rels


def load_model():
    ## donwload models
    dl.download_model('parsbert', process_func=dl._unzip_process_func)
    dl.download_model('dependencyparser')
    
    config = get_config()
    
    prefix = str(Path(__file__).parent.absolute()).replace('models', '')
    config['target_dir'] = prefix + config['target_dir']
    config['embeddings']['BertEmbeddings-0']['bert_model_or_path'] = prefix + config['embeddings-saved-dir']
    
    student=create_model(config)
    base_path=Path(config['target_dir'])/config['model_name']
    
    return student


def depparser(model, tokens_list):
    preds_arcs, preds_rels = model.predict(tokens_list,prediction_mode=True)
    preds_arcs = [p.item() for p in preds_arcs]
    return preds_arcs, preds_rels




