from pprint import pprint

from dadmatools.embeddings import get_embedding, get_all_embeddings_info, get_embedding_info

if __name__ == '__main__':
    # word_embeddings = get_embedding('fasttext-commoncrawl-vec')
    pprint(get_all_embeddings_info())
    # pprint(get_embedding_info('glodve-wiki'))
    word_embeddings = get_embedding('glove-wiki')
    word_embeddings.get_vocab()
    print(word_embeddings.word_vector('سلام'))
    print(word_embeddings.doesnt_match("رفتم کتاب بخرم"))
    print(word_embeddings.similarity('کتب', 'کتاب'))
    print(word_embeddings.embedding_text('امروز هوای خوبی بود'))