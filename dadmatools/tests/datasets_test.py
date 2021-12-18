from datasets import ARMAN
from datasets import get_datasets_info
from dadmatools.datasets import TEP
from dadmatools.datasets import PerSentLexicon
from dadmatools.datasets import FaSpell
from dadmatools.datasets import WikipediaCorpus
if __name__ == '__main__':
    print(get_datasets_info())
    print(get_datasets_info(tasks=['NER']))
    arman_dataset = ARMAN()
    print(len(arman_dataset))
    print(next(arman_dataset['train']))
    print(next(arman_dataset['test']))
    tep_dataset = TEP()
    print(len(tep_dataset))
    print (next(tep_dataset))
    persent = PerSentLexicon()
    print(len(persent))
    print (next(persent))
    faspell = FaSpell()
    print(len(faspell))
    print (next(faspell['faspell_main']))
    wikipedia_corpus = WikipediaCorpus()
    print(len(wikipedia_corpus))
    print (next(wikipedia_corpus))
