import os

from datasets.base import BaseDataset, DatasetInfo
from datasets.dataset_utils import download_with_progress, unzip_archive

URL = 'https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-1547/allzip'
DATASET_NAME = "FASpell"

def FaSpell(root):
    root = os.path.join(root, DATASET_NAME)

    def get_faspell_item(dir_addr, fname):
        f = open(os.path.join(dir_addr, fname))
        for line in f:
            try:
                correct, incorrect = line.split('\t')
            except:
                correct, incorrect, _ = line.split('\t')
            yield {'correct': correct, 'incorrect': incorrect}


    downloaded_file = download_with_progress(URL, root)
    dir_addr = unzip_archive(downloaded_file, root)
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.json')
    info = DatasetInfo(info_addr=info_addr)
    fa_spell_main = get_faspell_item(dir_addr, 'faspell_main.txt')
    fa_spell_ocr = get_faspell_item(dir_addr, 'faspell_ocr.txt')
    fa_spell_main = BaseDataset(fa_spell_main, info)
    fa_spell_ocr = BaseDataset(fa_spell_ocr, info)
    return {'faspell_main': fa_spell_main, 'faspell_ocr': fa_spell_ocr}
