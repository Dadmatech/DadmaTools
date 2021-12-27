import json
from enum import Enum
from pathlib import Path
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import fasttext
import numpy as np
import os
from dadmatools.embeddings.embedding_utils import download_with_progress, unzip_archive

EMBEDDINGS_INFO_ADDR = os.path.join(os.path.dirname(__file__), 'available_models.py')
EMBEDDINGS_INFO = json.load(open(EMBEDDINGS_INFO_ADDR))
DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), '.dadmatools', 'embeddings')

class EmbeddingType(Enum):
    FASTTEXT_BIN = 1
    KeyedVector = 2
    GLOVE = 3

def get_all_embeddings_info():
    return EMBEDDINGS_INFO


def get_embedding_info(emb_name):
    if emb_name not in EMBEDDINGS_INFO:
        raise KeyError(f'{emb_name} not exist! for see all supported embeddings call get_all_embeddings_info()')
    return EMBEDDINGS_INFO[emb_name]


def get_embedding(emb_name):
    if emb_name not in EMBEDDINGS_INFO:
        raise KeyError(f'{emb_name} not exist! for see all supported embeddings call get_all_embeddings_info()')
    alg = EMBEDDINGS_INFO[emb_name]['algorithm']
    format = EMBEDDINGS_INFO[emb_name]['format']
    if alg == 'fasttext' and format == 'bin':
        emb_type = EmbeddingType.FASTTEXT_BIN
    elif alg == 'glove':
        emb_type = EmbeddingType.GLOVE
    else:
        emb_type = EmbeddingType.KeyedVector
    url = EMBEDDINGS_INFO[emb_name]["url"]
    dest_dir = os.path.join(DEFAULT_CACHE_DIR, emb_name)
    os.makedirs(dest_dir, exist_ok=True)
    f_addr = os.path.join(dest_dir, EMBEDDINGS_INFO[emb_name]["filename"])
    if not os.path.exists(f_addr):
        zipped_file_name = download_with_progress(url, dest_dir)
        _ = unzip_archive(zipped_file_name, dest_dir, EMBEDDINGS_INFO[emb_name]["filename"])
    if emb_type == EmbeddingType.KeyedVector:
        if EMBEDDINGS_INFO[emb_name]["format"] == 'bin':
          model = KeyedVectors.load_word2vec_format(f_addr, binary=True)
        else:
          model = KeyedVectors.load_word2vec_format(f_addr)
    elif emb_type == EmbeddingType.FASTTEXT_BIN:
        model = fasttext.load_model(f_addr)
    elif emb_type == EmbeddingType.GLOVE:
        word2vec_addr = str(f_addr) + '_word2vec_format.vec'
        if not os.path.exists(word2vec_addr):
            _ = glove2word2vec(f_addr, word2vec_addr)
        model = KeyedVectors.load_word2vec_format(word2vec_addr)
        emb_type = EmbeddingType.KeyedVector
    return Embedding(model, emb_type, EMBEDDINGS_INFO[emb_name]["dim"])

class Embedding:
    def __init__(self, emb_model, emb_type, emb_dim):
        self.model = emb_model
        self.emb_type = emb_type
        self.emb_dim = emb_dim

    def __getitem__(self, word):
        return self.model[word]
    def doesnt_match(self, txt):
        return self.model.doesnt_match(txt.split())

    def similarity(self, w1, w2):
        return self.model.similarity(w1, w2)

    def embedding_text(self, text):
        if self.emb_type == EmbeddingType.FASTTEXT_BIN:
            return self.model.get_sentence_vector(text)
        if self.emb_type == EmbeddingType.KeyedVector:
            # try:
            #     index2word_set = set(self.model.wv.index2word)
            # except AttributeError:
            #     index2word_set = set(self.model.index_to_key)
            words = text.split()
            num_features = self.emb_dim
            feature_vec = np.zeros((num_features,), dtype='float32')
            n_words = 0
            for word in words:
                # if word in index2word_set:
                try:
                    feature_vec = np.add(feature_vec, self.model[word])
                    n_words += 1
                except:
                    pass
            if (n_words > 0):
                feature_vec = np.divide(feature_vec, n_words)
            return feature_vec

    def get_vocab(self):
        if self.emb_type == EmbeddingType.KeyedVector:
            try:
                return self.model.index_to_key
            except AttributeError:
                return self.model.wv
        else:
            return self.model.get_words(include_freq=True)

    def word_vector(self, word_name):
        if self.emb_type == EmbeddingType.KeyedVector:
            try:
                return self.model.wv.get_vector(word_name)
            except AttributeError:
                return self.model[word_name]
        if self.emb_type == EmbeddingType.FASTTEXT_BIN:
            return self.model.get_word_vector(word_name)

    def top_nearest(self, word, k):
        if self.emb_type == EmbeddingType.FASTTEXT_BIN:
            return self.model.get_nearest_neighbors(word, k)
        else:
            return self.model.most_similar(word, topn=k)