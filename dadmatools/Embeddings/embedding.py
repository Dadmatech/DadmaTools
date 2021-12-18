import json
from enum import Enum
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import fasttext
import numpy as np
import os
from embedding_utils import download_with_progress, unzip_archive

EMBEDDINGS_INFO = json.load(open('available_models.json'))
CACHE_DIR = './saved_embeddings/'
class EmbeddingType(Enum):
    FASTTEXT_BIN = 1
    KeyedVector = 2
    GLOVE = 3
# class ModelType(Enum):
#     FASTTEXT = 1
#     KeyedVector = 2

def get_embedding(emb_name):
    if emb_name not in EMBEDDINGS_INFO:
        raise KeyError('this embedding name not exist! please consider available_models.json')
    alg = EMBEDDINGS_INFO[emb_name]['algorithm']
    format = EMBEDDINGS_INFO[emb_name]['format']
    if alg == 'fasttext' and format == 'bin':
        emb_type = EmbeddingType.FASTTEXT_BIN
    elif alg == 'glove':
        emb_type = EmbeddingType.GLOVE
    else:
        emb_type = EmbeddingType.KeyedVector
    url = EMBEDDINGS_INFO[emb_name]["url"]
    dest_dir = os.path.join(CACHE_DIR, emb_name)
    os.makedirs(dest_dir, exist_ok=True)
    zipped_file_name = download_with_progress(url, dest_dir)
    _ = unzip_archive(zipped_file_name, dest_dir, EMBEDDINGS_INFO[emb_name]["filename"])
    f_addr = os.path.join(dest_dir, EMBEDDINGS_INFO[emb_name]["filename"])
    print(f_addr)
    if emb_type == EmbeddingType.KeyedVector:
        if EMBEDDINGS_INFO[emb_name]["format"] == 'bin':
          print('binary is true')
          model = KeyedVectors.load_word2vec_format(f_addr, binary=True)
        else:
          print('binary is false')
          model = KeyedVectors.load_word2vec_format(f_addr)
    elif emb_type == EmbeddingType.FASTTEXT_BIN:
        model = fasttext.load_model(f_addr)
    elif emb_type == EmbeddingType.GLOVE:
        word2vec_addr = str(f_addr) + '_word2vec_format.vec'
        _ = glove2word2vec(f_addr, word2vec_addr)
        model = KeyedVectors.load_word2vec_format(word2vec_addr)
        emb_type = EmbeddingType.KeyedVector
    return Embedding(model, emb_type, EMBEDDINGS_INFO[emb_name]["dim"])

class Embedding:
    def __init__(self, emb_model, emb_type, emb_dim):
        self.model = emb_model
        self.emb_type = emb_type
        self.emb_dim = emb_dim
    def doesnt_match(self, txt):
        return self.model.doesnt_match(txt.split())

    def similarity(self, w1, w2):
        return self.model.similarity(w1, w2)

    def get_text_embedding(self, text):
        if self.emb_type == EmbeddingType.FASTTEXT_BIN:
            return self.model.get_sentence_vector(text)
        if self.emb_type == EmbeddingType.KeyedVector:
            index2word_set = set(self.model.wv.index2word)
            words = text.split()
            num_features = self.emb_dim
            feature_vec = np.zeros((num_features,), dtype='float32')
            n_words = 0
            for word in words:
                if word in index2word_set:
                    n_words += 1
                    feature_vec = np.add(feature_vec, self.model[word])
            if (n_words > 0):
                feature_vec = np.divide(feature_vec, n_words)
            return feature_vec

    def get_vocab(self):
        return self.model.wv

    def get_vector_by_word_name(self, word_name):
        if self.emb_type == EmbeddingType.KeyedVector:
            return self.model.wv.get_vector(word_name)
        if self.emb_type == EmbeddingType.FASTTEXT_BIN:
            return self.model.get_word_vector(word_name)
    # def get_top_nearest(self, word, k):
    #     if self.emb_type == 'fasttext'
    #         model.get_nearest_neighbors('asparagus')
