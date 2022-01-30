from dadmatools.models.flair import embeddings as Embeddings
import copy
from dadmatools.models.flair.data import Dictionary
from dadmatools.models.flair import models as model
from pathlib import Path
from dadmatools.models.flair.list_data import ListCorpus
import torch
from pathlib import Path

import dadmatools.pipeline.download as dl

def get_config():
  config = {
    "Controller": {
      "model_structure": None
    },
    "ReinforcementTrainer": {
      "controller_learning_rate": 0.1,
      "controller_optimizer": "SGD",
      "distill_mode": False,
      "optimizer": "SGD",
      "sentence_level_batch": True
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
      "FastSequenceTagger": {
        "crf_attention": False,
        "dropout": 0.0,
        "hidden_size": 800,
        "sentence_loss": True,
        "use_crf": True
      }
    },
  #   "model_name": "xlmr-task_en-elmo_en-bert-task_multi-bert-task_word_en-flair_mflair_char_30episode_300epoch_32batch_0.1lr_800hidden_tweebank_monolingual_crf_fast_sqrtreward_reinforce_freeze_norelearn_sentbatch_0.5discount_5patience_nodev_new_upos8",
  #   "target_dir": "saved_models/",
      "model_name": "postagger.pt",
    "target_dir": "saved_models/postagger/",
    "targets": "upos",
    "upos": {
      "Corpus": "UniversalDependenciesCorpus-1",
      "UniversalDependenciesCorpus-1": {
        "data_folder": "flair/datasets/persian_dp"
      }
    },
    "train": {
      "controller_momentum": 0.9,
      "discount": 0.5,
      "learning_rate": 0.1,
      "max_episodes": 30,
      "max_epochs": 300,
      "max_epochs_without_improvement": 25,
      "mini_batch_size": 32,
      "monitor_test": False,
      "patience": 5,
      "save_final_model": False,
      "sqrt_reward": True,
      "train_with_dev": False,
      "true_reshuffle": False
    },
    "trainer": "ReinforcementTrainer"
  }
  return config


def make_tag_dictionary() -> Dictionary:
        # Make the tag dictionary
        tag_dictionary: Dictionary = Dictionary()
        tag_dictionary.add_item("O")
        # list = ["<unk>", "root", "ADJ", "NOUN", "ADP", "VERB", "PUNCT", "SCONJ", "AUX", "PART", "NUM", "CCONJ", "ADV", "DET", "PRON", "INTJ", "X"]
        list = ['X', 'NOUN', 'PRON', 'PUNCT', 'ADP', 'NUM', 'ADV', 'SCONJ', 'PART', 'PROPN', 'CCONJ', 'ADJ', 'INTJ', 'VERB', 'AUX', 'DET', '<unk>']
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
		kwargs['tag_type']=config['targets']
		kwargs['tag_dictionary']=make_tag_dictionary()
		tagger = getattr(model,classname)(**kwargs, config=config)
		tagger.word_map = None
		tagger.char_map = None
		tagger.lemma_map = None
		tagger.postag_map = None
        
		base_path=Path(config['target_dir'])/config['model_name']

		if (base_path / "best-model.pt").exists():
# 			print('Loading pretraining best model')
			tagger = tagger.load(base_path / "best-model.pt")
# 			torch.save({'state_dict' : tagger.state_dict(), 'use_se':True},'saved_models/pos/best-model.pt')
# 			tagger.save('saved_models/pos/best-model.pt')
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

def postagger_model(tokens_list):
    student=create_model(config)
    base_path=Path(config['target_dir'])/config['model_name']
    preds = student.predict(tokens_list, embeddings_storage_mode="none",prediction_mode=True)
    preds = [p.value for p in preds[0]]## removing the score for each token tag prediction
    return preds


# print(postagger_model(['علی', 'اول', 'مهر', 'مدرسه', 'رفت', '.']))

def load_model():
    ## donwload models
    dl.download_model('parsbert', process_func=dl._unzip_process_func)
    dl.download_model('postagger')
    
    config = get_config()

    prefix = str(Path(__file__).parent.absolute()).replace('models', '')
    config['target_dir'] = prefix + config['target_dir']
    config['embeddings']['BertEmbeddings-0']['bert_model_or_path'] = prefix + config['embeddings-saved-dir']
    
    student=create_model(config)
    # base_path=Path(config['target_dir'])/config['model_name']
    
    return student

def postagger(model, tokens_list):
    preds = model.predict(tokens_list, embeddings_storage_mode="none",prediction_mode=True)
    preds = [p.value for p in preds[0]]## removing the score for each token tag prediction
    return preds

