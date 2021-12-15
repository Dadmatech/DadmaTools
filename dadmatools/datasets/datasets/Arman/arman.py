import os

from datasets.base import BaseDataset, SplittedDataset, DatasetInfo
from datasets.dataset_utils import download_with_progress, unzip_archive, load_dataset_info

URL = 'https://raw.githubusercontent.com/HaniehP/PersianNER/master/ArmanPersoNERCorpus.zip'
DATASET_NAME = "ARMAN"
def ARMAN(root):
    root = os.path.join(root, DATASET_NAME)

    def get_arman_item(dir_addr, split):
        f_addr = os.path.join(dir_addr, split)
        f = open(f_addr)
        sentence = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    yield sentence
                    sentence = []
                continue
            splits = line.split(' ')
            sentence.append([splits[0], splits[-1].rstrip("\n")])

        if len(sentence) > 0:
            yield sentence
            sentence = []


    downloaded_file = download_with_progress(URL, root)
    dir_addr = unzip_archive(downloaded_file, root)

    train_iterator = get_arman_item(dir_addr, 'train_fold1.txt')
    test_dataset = get_arman_item(dir_addr, 'test_fold1.txt')
    # dev_iterator = get_arman_item(dir_addr, 'dev')
    # train = BaseDataset(train_iterator)
    # test = BaseDataset(test_iterator)
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.json')
    info = DatasetInfo(info_addr=info_addr)
    train_dataset = BaseDataset(train_iterator,info)
    test_dataset = BaseDataset(test_dataset,info)
    return {'train': train_dataset, 'test': test_dataset}
