import json
import os
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, unzip_dataset, is_exist_dataset, DEFAULT_CACHE_DIR

URL = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3195/lscp-0.5-fa-normalized.7z'
DATASET_NAME = "PersianTweets"


def PersianTweets(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.py')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_tweets(dest_dir):
        addr = os.path.join(dest_dir, "lscp-0.5-fa-normalized.txt")
        f = open(addr, "r")
        for line in f:
            yield line.strip()

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        downloaded_file = download_dataset(URL, dest_dir)
        dest_dir = unzip_dataset(downloaded_file, dest_dir, zip_format='7z')
    info = DatasetInfo(info_addr=info_addr)
    size = DATASET_INFO['size']
    tweets_iterator = BaseIterator(get_tweets(dest_dir), num_lines=size)
    dataset = BaseDataset(info)
    dataset.set_iterators(tweets_iterator)
    return dataset
