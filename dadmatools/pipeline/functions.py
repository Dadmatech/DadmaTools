import spacy
from spacy import displacy
from spacy.language import Language
from spacy.tokens import Doc, Token, Span
from spacy.pipeline import Sentencizer

import dadmatools.models.tokenizer as tokenizer
import dadmatools.models.lemmatizer as lemmatizer
import dadmatools.models.postagger as tagger
import dadmatools.models.dependancy_parser as dp


nlp = spacy.blank('fa')
Token.set_extension("dep_arc", default=None)
Doc.set_extension("sentences", default=None)
sentencizer = Sentencizer()

## function for tokenizer
@Language.component('tokenizer', retokenizes=True)
def tokenize(doc):
    
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[0:len(doc)])
    starts = []
    tokens_list = tokenizer.tokenize(doc.text)
    tokens = []
    index = 0
    for l in tokens_list:
        starts.append(index)
        for t in l: 
            tokens.append(t)
            index += 1
    doc = Doc(nlp.vocab, words=tokens)
    spans = []
    for idx, i in enumerate(starts):
        if idx+1 == len(starts):
            spans.append(Span(doc, i, index))
        else:
            spans.append(Span(doc, i, starts[idx+1]))
    doc._.sentences = spans
        
    return doc


## function for lemmatizer
@Language.component('lemmatize', assigns=["token.lemma"])
def lemma(doc):
    for sent in doc._.sentences:
        tokens = [d.text for d in sent]
        print(tokens)
        lemmas = lemmatizer.lemmatize([tokens])
        for idx, d in enumerate(sent): d.lemma_ = lemmas[idx]
        
    return doc


## function for lemmatizer
@Language.component('postagger', assigns=["token.pos"])
def postagger(doc):
    for sent in doc._.sentences:
        tokens = [d.text for d in sent]
        tags = tagger.postagger_model(tokens)

        for idx, d in enumerate(sent): d.pos_ = tags[idx]

    return doc


## function for lemmatizer
@Language.component('dependancyparser', assigns=["token.dep"])
def depparser(doc):
    
    for sent in doc._.sentences:
        tokens = [d.text for d in sent]
        preds_arcs, preds_rels = dp.dependancy_parser_model(tokens)

        for idx, d in enumerate(sent):
            arc = preds_arcs[idx]
            rel = preds_rels[idx]
            d.dep_ = rel
            d._.dep_arc = arc
            d.head = sent[arc-1]
    
    return doc

def to_json(doc):
    dict_list = []
    for sent in doc._.sentences:
        sentence = []
        for i, d in enumerate(sent):
            dictionary = {}
            dictionary['id'] = i+1
            dictionary['text'] = d.text
            dictionary['lemma'] = d.lemma_
            dictionary['pos'] = d.pos_
            dictionary['rel'] = d.dep_
            dictionary['root'] = d._.dep_arc
            sentence.append(dictionary)
        dict_list.append(sentence)
    return dict_list

nlp.add_pipe('tokenizer', first=True)
# nlp.add_pipe('sentencizer', after='tokenizer')
nlp.add_pipe('lemmatize', after='tokenizer')
nlp.add_pipe('postagger', after='lemmatize')
nlp.add_pipe('dependancyparser', after='postagger')

from spacy.lang.fa import Persian
nlp = Persian().from_disk('pipline/farsi-kit')
Doc.set_extension("sentences", default=None)


# lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
# nlp = lang_cls.from_config(config)
# nlp.from_bytes(bytes_data)

# config = nlp.config
# bytes_data = nlp.to_bytes()
# nlp.to_disk('pipline/farsi-kit')    
    
print(nlp.pipe_names)
pipe_analysis = nlp.analyze_pipes(pretty=True)
print(pipe_analysis)
doc = nlp('من دیروز به کتابخانه رفتم! فردا به مدرسه می‌روم.')
print(to_json(doc))

# for i in range(len(doc)): # leftward immediate children of the word in the syntactic dependency parse.
#     print([t.text for t in doc[i].lefts])
# chunks = list(doc.noun_chunks)
# print(chunks)

print(list(doc._.sentences))
print(len(list(doc._.sentences)))
# displacy.render(doc.sents[0], style='dep')
