import json
import os
from wiki_dump_reader import Cleaner, iterate
from dadmatools.datasets.base import BaseDataset, DatasetInfo
from dadmatools.datasets.dataset_utils import download_dataset, unzip_dataset, is_exist_dataset, DEFAULT_CACHE_DIR

URL = 'https://dumps.wikimedia.org/fawiki/20211201/fawiki-20211201-pages-meta-current.xml.bz2'
DATASET_NAME = "wikipedia"
DESC = 'fawiki dump progress on 20211201 / All pages, current versions only.'
def WikipediaCorpus(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.json')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_wikipedia_item(dest_dir):
        extracted_f = os.path.join(dest_dir, "fawiki-20211201-pages-meta-current.xml")
        cleaner = Cleaner()
        for title, text in iterate(extracted_f):
            text = cleaner.clean_text(text)
            cleaned_text, links = cleaner.build_links(text)
            yield {'title': title, 'content': cleaned_text, 'links': links}

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        downloaded_file = download_dataset(URL, dest_dir)
        dest_dir = unzip_dataset(downloaded_file, dest_dir)
    info = DatasetInfo(info_addr=info_addr)
    wiki_iterator = get_wikipedia_item(dest_dir)
    return BaseDataset(iterator=wiki_iterator, info=info)
