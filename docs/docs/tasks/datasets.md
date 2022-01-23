Datasets
========

This section keeps a list of Persian NLP datasets publicly available. 

| Dataset                                                | Task                     | 
|--------------------------------------------------------|--------------------------|
|PersianNER                                              | Named Entity Recognition  |
|ARMAN                                              | Named Entity Recognition  |
|Peyma                                              | Named Entity Recognition  |
|FarsTail                                              | Textual Entailment  |
|FaSpell                                              | Spell Checking  |
|PersianNews                                              | Text Classification  |
|PerUDT                                              | Universal Dependency  |
|PnSummary                                              | Text Summarization  |
|SnappfoodSentiment                                              | Sentiment Classification  |
|TEP                                              | Text Translation(eng-fa)  |
|WikipediaCorpus                                              | Corpus |
|PersianTweets                                              | Corpus  |

We Will add the description of all datasets in the future.

```python
from dadmatools.datasets import FarsTail
from dadmatools.datasets import SnappfoodSentiment
from dadmatools.datasets import Peyma
from dadmatools.datasets import PerUDT
from dadmatools.datasets import PersianTweets
from dadmatools.datasets import PnSummary


farstail = FarsTail()
#len of dataset
print(len(farstail.train))

#like a generator
print(next(farstail.train))

#dataset details
pn_summary = PnSummary()
print('PnSummary dataset information: ', pn_summary.info)

#loop over dataset
snpfood_sa = SnappfoodSentiment()
for i, item in enumerate(snpfood_sa.test):
    print(item['comment'], item['label'])

#get first tokens' lemma of all dev items
perudt = PerUDT()
for token_list in perudt.dev:
    print(token_list[0]['lemma'])

#get NER tag of first Peyma's data
peyma = Peyma()
print(next(peyma.data)[0]['tag'])

#corpus 
tweets = PersianTweets()
print('tweets count : ', len(tweets.data))
print('sample tweet: ', next(tweets.data))
```

