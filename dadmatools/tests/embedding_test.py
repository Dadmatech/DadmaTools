from dadmatools.embeddings import get_embedding, get_all_embeddings_info, get_embedding_info

if __name__ == '__main__':
    # word_embeddings = get_embedding('fasttext-commoncrawl-vec')
    print(get_all_embeddings_info())
    print(get_embedding_info('glove-wiki'))
    word_embeddings = get_embedding('glove-wiki')
    word_embeddings.get_vocab()
    print(word_embeddings.get_vector_by_word_name('سلام'))
    print(word_embeddings.doesnt_match("رفتم کتاب بخرم"))
    print(word_embeddings.similarity('دفتر', 'کتاب'))
    print(word_embeddings.get_text_embedding('امروز هوای خوبی بود'))