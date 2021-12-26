from pprint import pprint
from dadmatools.embeddings import get_embedding, get_all_embeddings_info, get_embedding_info

def test_embedding():
    pprint(get_all_embeddings_info())
    embedding = get_embedding('glove-wiki')
    print(embedding['سلام'])



