Pretrained Persian embeddings
=============================

This repository keeps a list of pretrained word embeddings publicly available in Persian. The `dadmatools.embeddings` provides functions for using the embeddings as well as using common functions dealing with them.

| Name                                                         | Embedding Algorithm    | Corpus    | 
| ------------------------------------------------------------ | ---------------------- | ------    |
| [glove-wiki](https://github.com/Text-Mining/Persian-Wikipedia-Corpus/tree/master/models/glove)                                               | glove                  | Wikipedia | 
| [fasttext-commoncrawl-bin](https://fasttext.cc/docs/en/crawl-vectors.html)                                 | fasttext               | CommonCrawl |
| [fasttext-commoncrawl-vec](https://fasttext.cc/docs/en/crawl-vectors.html)                                 | fasttext            | CommonCrawl | 
| [word2vec-conll](http://vectors.nlpl.eu/)                                   | word2vec          | Persian CoNLL17 corpus | 

Embeddings are a way of representing text as numeric vectors, and can be calculated both for chars, subword units, words, sentences or documents. There Persian word embedding models can be used easily using DadmaTools.

```python
from dadmatools.embeddings import get_embedding, get_all_embeddings_info, get_embedding_info
from pprint import pprint

pprint(get_all_embeddings_info())

#get embedding information of specific embedding
embedding_info = get_embedding_info('glove-wiki')

#### load embedding ####
word_embedding = get_embedding('glove-wiki')

#get vector of the word
print(word_embedding['سلام'])

#vocab
vocab = word_embedding.get_vocab()

### some useful functions ###
print(word_embedding.top_nearest("زمستان", 10))
print(word_embedding.similarity('کتب', 'کتاب'))
print(word_embedding.embedding_text('امروز هوای خوبی بود'))
```

