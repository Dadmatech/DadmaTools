import os

from dadmatools.datasets.base import BaseDataset, DatasetInfo
from dadmatools.datasets.dataset_utils import download_dataset, unzip_dataset

URL = 'https://ds.kplab.ir/s/tJ6d8Mifm8eeQ44/download/ctree-khajeh-nasir.tar.xz'
BASE_PATH = 'Train_Test_Dev_implemented_from_SAZEH_trees'
DATASET_NAME = "UPDT"

def UPDT(root):
    root = os.path.join(root, DATASET_NAME)

    def get_updt_item(dir_addr, fname):
        f = open(os.path.join(dir_addr, BASE_PATH, fname))
        for line in f:
            yield line

    downloaded_file = download_with_progress(URL, root)
    dir_addr = unzip_archive(downloaded_file, root)

    train = get_updt_item(dir_addr, 'train.txt')
    test = get_updt_item(dir_addr, 'test.txt')
    dev = get_updt_item(dir_addr, 'dev.txt')
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.json')
    info = DatasetInfo(info_addr=info_addr)
    train = BaseDataset(train, info)
    test = BaseDataset(test, info)
    dev = BaseDataset(dev, info)
    return {'train': train, 'test': test, 'dev':dev}
