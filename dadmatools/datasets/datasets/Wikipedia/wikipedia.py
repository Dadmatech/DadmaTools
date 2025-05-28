import json
import os
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, unzip_dataset, is_exist_dataset, DEFAULT_CACHE_DIR

URL = 'https://drive.google.com/uc?id=1jHje8Q07tQWEpt8cEpFR_TOuqjFs79Vb'
DATASET_NAME = "WikipediaCorpus"

def WikipediaCorpus(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.py')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_wikipedia_item(dest_dir):
        addr = os.path.join(dest_dir, "cleaned_wiki.txt")
        f = open(addr, 'r',encoding='utf-8')
        for line in f:
            yield json.loads(line)

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        downloaded_file = os.path.join(dest_dir, 'wikipedia.tar.xz')
        download_dataset(URL, dest_dir, filename=downloaded_file)
        dest_dir = unzip_dataset(downloaded_file, dest_dir, zip_format='xz')
    info = DatasetInfo(info_addr=info_addr)
    wiki_iterator = BaseIterator(get_wikipedia_item(dest_dir), num_lines=DATASET_INFO['size'])
    dataset = BaseDataset(info=info)
    dataset.set_iterators(wiki_iterator)
    return dataset

