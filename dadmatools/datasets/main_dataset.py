from datasets.dataset_utils import get_datasets_info
from datasets.datasets.Arman.arman import ARMAN
from datasets.datasets.FaSpell.sp import FaSpell
from datasets.datasets.TEP.tep import TEP
from datasets.datasets.UPDT.updt import UPDT
from datasets.datasets.Wikipedia.wikipedia import WikipediaCorpus
from datasets.datasets.persent.persent import PerSentLexicon
CACHE_DIR = './saved_datasets/'
if __name__ == '__main__':
    print(get_datasets_info())
    print(get_datasets_info(tasks=['NER']))
    arman_dataset = ARMAN(CACHE_DIR)
    print(len(arman_dataset))
    print(next(arman_dataset['train']))
    print(next(arman_dataset['test']))
    tep_dataset = TEP(CACHE_DIR)
    print(len(tep_dataset))
    print (next(tep_dataset))
    persent = PerSentLexicon(CACHE_DIR)
    print(len(persent))
    print (next(persent))
    faspell = FaSpell(root='./')
    print(len(faspell))
    print (next(faspell['faspell_main']))
    wikipedia_corpus = WikipediaCorpus(CACHE_DIR)
    print(len(wikipedia_corpus))
    print (next(wikipedia_corpus))
