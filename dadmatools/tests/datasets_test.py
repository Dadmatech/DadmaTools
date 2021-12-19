from dadmatools.datasets import ARMAN
from dadmatools.datasets import get_all_datasets_info, get_dataset_info
from dadmatools.datasets import TEP
from dadmatools.datasets import PerSentLexicon
from dadmatools.datasets import FaSpell
from dadmatools.datasets import WikipediaCorpus
if __name__ == '__main__':
    print(get_dataset_info('TP'))
    print(get_all_datasets_info(tasks=['NER']))
    arman_dataset = ARMAN()
    print(len(arman_dataset['train']))
    print(len(arman_dataset['test']))
    # print(next(arman_dataset['train']))
    # for i, item in enumerate(arman_dataset['train']):
    #     print('train arman', i)
    # print('len arman train ', i )
    # print(next(arman_dataset['test']))
    # for i, item in enumerate(arman_dataset['test']):
    #     pass
    # print('len arman test ', i )
    #
    # tep_dataset = TEP()
    # for i, item in enumerate(tep_dataset):
    #     pass
    # print('len tep ', i)
    # print(len(tep_dataset))
    # print (next(tep_dataset))
    persent = PerSentLexicon()
    # for i, item in enumerate(persent):
    #     pass
    # print('len persent ', i)
    print(len(persent))
    # print (next(persent))
    faspell = FaSpell()
    # for i, item in enumerate(faspell['faspell_main']):
    #     pass
    # print('len faspell main ', i)
    # for i, item in enumerate(faspell['faspell_ocr']):
    #     pass
    # print('len faspell ocr ', i)
    print(len(faspell['faspell_main']))
    print(len(faspell['faspell_ocr']))
    # print (next(faspell['faspell_main']))
    # wikipedia_corpus = WikipediaCorpus()
    # print(len(wikipedia_corpus))
    # print (next(wikipedia_corpus))
