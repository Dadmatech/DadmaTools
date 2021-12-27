import numpy
from numpy import dot
from numpy.linalg import norm
from dadmatools.embeddings import get_embedding, get_all_embeddings_info

def test_embedding():
    embedding = get_embedding('glove-wiki')
    assert type(embedding['کلمه']) == numpy.ndarray


def test_info():
    assert(len(get_all_embeddings_info())) == 4


def test_similarity():
    embedding = get_embedding('glove-wiki')
    assert embedding.similarity('پدر', 'مادر') > embedding.similarity('پدر', 'باجناق')


def test_text_embedding():
    cosine = lambda vec1,vec2: dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    embedding = get_embedding('glove-wiki')
    txt1_vec = embedding.embedding_text('تیم ملی فوتبال ایران')
    txt2_vec = embedding.embedding_text('علی کریمی')
    txt3_vec = embedding.embedding_text('تیم ملی والیبال ایران')
    assert cosine(txt1_vec, txt2_vec) > cosine(txt2_vec, txt3_vec)