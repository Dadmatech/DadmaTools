from dadmatools.pipeline import Pipeline
from dadmatools.pipeline.persian_tokenization.tokenizer import tokenizer, load_tokenizer_model

nlp = Pipeline('lem,pos,ner,dep,cons,spellchecker,kasreh,sent')

text = 'اینو اگه خواستین میتونین تست کنین واسه تبدیل'
doc = nlp(text)
print(doc)