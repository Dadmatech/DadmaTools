import os
from wiki_dump_reader import Cleaner, iterate
from datasets.base import BaseDataset, DatasetInfo
from datasets.dataset_utils import download_with_progress, unzip_archive

URL = 'https://dumps.wikimedia.org/fawiki/20211201/fawiki-20211201-pages-meta-current.xml.bz2'
DATASET_NAME = "wikipedia"
DESC = 'fawiki dump progress on 20211201 / All pages, current versions only.'
def WikipediaCorpus(root):
    root = os.path.join(root, DATASET_NAME)

    def get_wikipedia_item(extracted_f):
        cleaner = Cleaner()
        for title, text in iterate(extracted_f):
            text = cleaner.clean_text(text)
            cleaned_text, links = cleaner.build_links(text)
            yield {'title': title, 'content': cleaned_text, 'links': links}

    downloaded_file = download_with_progress(URL, root)
    extracted_f = unzip_archive(downloaded_file, root)
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.json')
    info = DatasetInfo(info_addr=info_addr)
    wiki_iterator = get_wikipedia_item(extracted_f)
    return BaseDataset(iterator=wiki_iterator, info=info)
