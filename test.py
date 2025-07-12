from dadmatools.normalizer import Normalizer
import os
# ⚠️ Fixes protobuf+sentencepiece descriptor errors across environments
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Optional: verify the backend
try:
    from google.protobuf.internal import api_implementation
    print("Protobuf backend:", api_implementation.Type())
except Exception as e:
    print("Protobuf status check:", e)

normalizer = Normalizer(
    full_cleaning=True,
    unify_chars=True,
    refine_punc_spacing=True,
    remove_extra_space=True,
    remove_puncs=True,
    remove_html=True,
    remove_stop_word=True,
    replace_email_with="<EMAIL>",
    replace_number_with="YNWA",
    replace_url_with="HAFEZ",
    replace_mobile_number_with="MOBILE",
    replace_emoji_with="EMOJIS",
    replace_home_number_with="KOOSHI"
)

text = """
<p>
1404/04/02,, عالیه,  کوشی     و درسته
44433515
02144433515
091900732654
:) (: ، ایا یا شما
mo.erfan1379@gmail.com
+9844433515
دادماتولز اولین نسخش سال ۱۴۰۰ منتشر شده.
امیدواریم که این تولز بتونه کار با متن رو براتون شیرین‌تر و راحت‌تر کنه
لطفا با ایمیل dadmatools@dadmatech.ir با ما در ارتباط باشید
آدرس گیت‌هاب هم که خب معرف حضور مبارک هست:
 https://github.com/Dadmatech/DadmaTools
</p>
"""
print('input text : ', text)
print('output text when replace emails and remove urls : ', normalizer.normalize(text))

# #full cleaning
normalizer = Normalizer(full_cleaning=True)
print('output text when using full_cleaning parameter', normalizer.normalize(text))

# from dadmatools.pipeline.informal2formal.main import Informal2Formal
# translator = Informal2Formal()

# print(translator.translate('اینو اگه خواستین میتونین واسه تبدیل تست کنین '))


import dadmatools.pipeline.language as language

# as tokenizer is the default tool, it will be loaded even without calling
pips = 'tok,lem,pos,dep,chunk,cons,spellchecker,kasreh,itf,ner,sent'
nlp = language.Pipeline(pips)


text = 'من صادق جعفری‌زاده به عنوان توسعه‌دهنده دادماتولز از شرکت دادماتک هستم. من به لوزامبورگ خواهم رفت.'
doc = nlp(text)
print(doc)


from dadmatools.datasets import get_all_datasets_info, get_dataset_info
from dadmatools.datasets import ARMAN
from dadmatools.datasets import TEP
from dadmatools.datasets import PerSentLexicon
from dadmatools.datasets import FaSpell
from dadmatools.datasets import WikipediaCorpus
from dadmatools.datasets import PersianNer
from dadmatools.datasets import PersianNews
from dadmatools.datasets import PnSummary
from dadmatools.datasets import FarsTail
from dadmatools.datasets import SnappfoodSentiment
from dadmatools.datasets import get_all_datasets_info
from dadmatools.datasets import Peyma
from dadmatools.datasets import PerUDT
from dadmatools.datasets import PersianTweets
from pprint import pprint

pprint(get_all_datasets_info(tasks=['NER', 'Sentiment-Analysis']))

pprint(get_dataset_info('PerUDT'))

print('*** WikipediaCorpus dataset ****')
print()
wiki = WikipediaCorpus()
print('len data ', len(wiki.data))
print()
print('sample: ', next(wiki.data))
print()
print('****** dataset details:********\n ')
print(wiki.info)
arman = ARMAN()
print('**** Arman dataset **** ')
print('splits: ', arman.info.splits)
print(len(arman.train))
print(next(arman.test))

from dadmatools.embeddings import get_embedding, get_all_embeddings_info, get_embedding_info
pprint(get_all_embeddings_info())
pprint(get_embedding_info('glove-wiki'))
embedding = get_embedding('glove-wiki')
print(embedding['ابزار'])
print(embedding.embedding_text('ابزار پردازش متن فارسی'))
embedding.similarity('کتاب', 'کتب')
embedding.top_nearest(embedding['کتاب'], 10)
embedding.top_nearest('کتاب', 10)