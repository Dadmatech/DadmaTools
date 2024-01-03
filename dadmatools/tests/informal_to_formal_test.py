from utils import set_root_dir

set_root_dir()

from dadmatools.pipeline import Pipeline

nlp = Pipeline('lem,pos,ner,dep,cons,spellchecker,kasreh,sent')

text = 'اینو اگه خواستین میتونین تست کنین واسه تبدیل'
doc = nlp(text)
print(doc)