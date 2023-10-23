Quick start
===========

Once you have [installed](installation.md) the DadmaTools package, you can use it in your python project using `import DadmaTools`. 

You will find the main functions through the `models` and `datasets` modules -- see the library documentation for more details about how to use the different functions for loading models and datasets. 
For analysing texts in Persian, you will primarily need to import functions from `dadmatools.pipeline.language` in order to load and use our pipeline. 

The DadmaTools package provides you with several models for different NLP tasks using different frameworks. 
On this section, you will have a quick tour of the main functions of the DadmaTools package. 
For a more detailed description of the tasks and frameworks, follow the links to the documentation: 

*  [Embedding of text](../tasks/embeddings.md) with Gensim and Fasttext
*  [Datasets](../tasks/datasets.md)
*  [Normalizing](../tasks/normalizing.md)
*  [Lemmatizing](../tasks/normalizing.md) with LSTM
*  [Part of speech tagging](../tasks/pos.md) (POS) with BERT
*  [Named Entity Recognition](../tasks/ner.md) (NER) with BERT
*  [Dependency parsing and NP-chunking](../tasks/dependency.md) with BERT
<!-- 
You can also try out our [getting started jupyter notebook](https://github.com/Dadmatech/dadmatools/blob/master/examples/tutorials/getting_started.ipynb) for quicly learning how to load and use the DadmaTools models and datasets.  -->

## All-in-one with the spaCy models

With DadmaTools you can try out different NLP tasks along with other pipelines that are already presented in spaCy. The main advantages of the spaCy model is that it is fast and it includes many functions based on NLP tasks which can be used easily. 

<!-- The main functions are:  

* `load_spacy_model` for loading a spaCy model for POS, NER and dependency parsing or a spaCy sentiment model
* `load_spacy_chunking_model` for loading a wrapper around the spaCy model with which you can deduce NP-chunks from dependency parses -->

## Pre-processing tasks

Perform [Part-of-Speech tagging](../tasks/pos.md), [Named Entity Recognition](../tasks/ner.md) and [dependency parsing](../tasks/dependency.md) at the same time with the DadmaTools spaCy model.
Here is a snippet to quickly getting started: 

For text normalizing you can use the `dadmatools.models.normalizer`. 

```python
from dadmatools.models.normalizer import Normalizer

normalizer = Normalizer(
    full_cleaning=False,
    unify_chars=True,
    refine_punc_spacing=True,
    remove_extra_space=True,
    remove_puncs=False,
    remove_html=False,
    remove_stop_word=False,
    replace_email_with="<EMAIL>",
    replace_number_with=None,
    replace_url_with="",
    replace_mobile_number_with=None,
    replace_emoji_with=None,
    replace_home_number_with=None
)

text = """
<p>
دادماتولز اولین نسخش سال ۱۴۰۰ منتشر شده. 
امیدواریم که این تولز بتونه کار با متن رو براتون شیرین‌تر و راحت‌تر کنه
لطفا با ایمیل dadmatools@dadmatech.ir با ما در ارتباط باشید
آدرس گیت‌هاب هم که خب معرف حضور مبارک هست:
 https://github.com/Dadmatech/DadmaTools
</p>
"""
normalized_text = normalizer.normalize(text)
#<p> دادماتولز اولین نسخش سال 1400 منتشر شده. امیدواریم که این تولز بتونه کار با متن رو براتون شیرین‌تر و راحت‌تر کنه لطفا با ایمیل <EMAIL> با ما در ارتباط باشید آدرس گیت‌هاب هم که خب معرف حضور مبارک هست: </p>

#full cleaning
normalizer = Normalizer(full_cleaning=True)
normalized_text = normalizer.normalize(text)
#دادماتولز نسخش سال منتشر تولز بتونه کار متن براتون شیرین‌تر راحت‌تر کنه ایمیل ارتباط آدرس گیت‌هاب معرف حضور مبارک

```


## Sequence labelling with BERT

For part-of-speech tagging, dependancy parsing, constituency parsing and named entity recognition, BERT models are presented.

```python
import dadmatools.pipeline_v1.language as language

# here lemmatizer and pos tagger will be loaded
# as tokenizer is the default tool, it will be loaded as well even without calling
pips = 'pos,dep,cons,ner'
nlp = language.Pipeline(pips)

# you can see the pipeline with this code
print(nlp.analyze_pipes(pretty=True))

# doc is an SpaCy object
doc = nlp('از قصهٔ کودکیشان که می‌گفت، گاهی حرص می‌خورد!')

dictionary = language.to_json(pips, doc)
print(dictionary)  ## to show pos tags, dependancy parses, and constituency parses
print(doc._.ners)

```


