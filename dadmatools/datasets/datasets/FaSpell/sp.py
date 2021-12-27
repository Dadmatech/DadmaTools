import json
import os

from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, unzip_dataset, is_exist_dataset, DEFAULT_CACHE_DIR

URL = 'https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-1547/allzip'
DATASET_NAME = "FaSpell"

def FaSpell(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.py')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_faspell_item(dir_addr, fname):
        f = open(os.path.join(dir_addr, fname))
        for i, line in enumerate(f):
            if i == 0:
                continue
            try:
                correct, incorrect = line.replace('\n', '').split('\t')
            except:
                incorrect, correct, _ = line.replace('\n', '').split('\t')
            yield {'correct': correct, 'wrong': incorrect}

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        downloaded_file = download_dataset(URL, dest_dir)
        dest_dir = unzip_dataset(downloaded_file, dest_dir)
    info = DatasetInfo(info_addr=info_addr)
    fa_spell_main = get_faspell_item(dest_dir, 'faspell_main.txt')
    fa_spell_ocr = get_faspell_item(dest_dir, 'faspell_ocr.txt')
    fa_spell_main_size = DATASET_INFO['size']['main']
    fa_spell_ocr_size = DATASET_INFO['size']['ocr']
    main_iterator = BaseIterator(fa_spell_main, num_lines=fa_spell_main_size)
    ocr_iterator =  BaseIterator(fa_spell_ocr, num_lines=fa_spell_ocr_size)
    iterators = {'main': main_iterator, 'ocr': ocr_iterator}
    faspell_dataset = BaseDataset(info)
    faspell_dataset.set_iterators(iterators)
    return faspell_dataset
