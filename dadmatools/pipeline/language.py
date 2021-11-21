import spacy
from spacy import displacy
from spacy.language import Language
from spacy.tokens import Doc, Token, Span
from spacy.pipeline import Sentencizer

import dadmatools.models.tokenizer as tokenizer
import dadmatools.models.lemmatizer as lemmatizer
import dadmatools.models.postagger as tagger
import dadmatools.models.dependancy_parser as dp
import dadmatools.models.constituency_parser as conspars


class NLP():
    """
    In this class a blank pipeline in created and it is initialized based on our trained models
    possible pipelines: [tokenizer, lemmatize, postagger, dependancyparser]
    """
    tokenizer_model = None
    lemma_model = None
    postagger_model = None
    depparser_model = None
    
    Token.set_extension("dep_arc", default=None)
    Doc.set_extension("sentences", default=None)
    Doc.set_extension("chunks", default=None)
    Doc.set_extension("constituency", default=None)
    
    global nlp
    nlp = None
    
    def __init__(self, lang, pipelines):
        
        global nlp
        nlp = spacy.blank(lang)
        self.nlp = nlp
        
        self.dict = {'tok':'tokenizer', 'lem':'lemmatize', 'pos':'postagger', 'dep':'dependancyparser', 'cons':'constituencyparser'}
        self.pipelines = pipelines.split(',')
        
        global tokenizer_model
        tokenizer_model = tokenizer.load_model()
        self.nlp.add_pipe('tokenizer', first=True)
        
        if 'lem' in pipelines:
            global lemma_model
            lemma_model = lemmatizer.load_model()
            self.nlp.add_pipe('lemmatize')
        
        if 'dep' in pipelines:
            global depparser_model
            depparser_model = dp.load_model()
            self.nlp.add_pipe('dependancyparser')
        
        if 'pos' in pipelines:
            global postagger_model
            postagger_model = tagger.load_model()
            self.nlp.add_pipe('postagger')
        
        if 'cons' in pipelines:
            global consparser_model
            consparser_model = conspars.load_model()
            self.nlp.add_pipe('constituencyparser')
    
    @Language.component('tokenizer', retokenizes=True)
    def tokenizer(doc):
        model, args = tokenizer_model
        
        with doc.retokenize() as retokenizer:
            retokenizer.merge(doc[0:len(doc)])
        starts = []
        tokens_list = tokenizer.tokenizer(model, args, doc.text)
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

    @Language.component('lemmatize', assigns=["token.lemma"])
    def lemmatizer(doc):
        
        model, args = lemma_model
        for sent in doc._.sentences:
            tokens = [d.text for d in sent]
            lemmas = lemmatizer.lemma(model, args, [tokens])
            for idx, d in enumerate(sent): d.lemma_ = lemmas[idx]
        
        return doc
    
    @Language.component('postagger', assigns=["token.pos"])
    def postagger(doc):
        model = postagger_model
        
        for sent in doc._.sentences:
            tokens = [d.text for d in sent]
            tags = tagger.postagger(model, tokens)
            
            for idx, d in enumerate(sent): d.pos_ = tags[idx]
        
        return doc
    
    @Language.component('dependancyparser', assigns=["token.dep"])
    def depparser(doc):
        model = depparser_model
        
        for sent in doc._.sentences:
            tokens = [d.text for d in sent]
            preds_arcs, preds_rels = dp.depparser(model, tokens)

            for idx, d in enumerate(sent):
                arc = preds_arcs[idx]
                rel = preds_rels[idx]
                d.dep_ = rel
                d._.dep_arc = arc
                d.head = sent[arc-1]
        
        return doc
    
    @Language.component('constituencyparser')
    def constituencyparser(doc):
        model = consparser_model
        
        constitu_parses = []
        chunks = []
        for sent in doc._.sentences:
            ## getting the constituency of the sentence
            cons_res = conspars.cons_parser(model, sent.text)
            constitu_parses.append(cons_res)
            ## getting the chunks of the sentence
            chunker_res = conspars.chunker(cons_res)
            chunks.append(chunker_res)
        
        doc._.constituency = constitu_parses
        doc._.chunks = chunks
        
        return doc
    
    
    def to_json(self, doc):
        dict_list = []
        for sent in doc._.sentences:
            sentence = []
            for i, d in enumerate(sent):
                dictionary = {}
                dictionary['id'] = i+1
                dictionary['text'] = d.text
                if 'lem' in self.pipelines: dictionary['lemma'] = d.lemma_
                if 'pos' in self.pipelines: dictionary['pos'] = d.pos_
                if 'dep' in self.pipelines: 
                    dictionary['rel'] = d.dep_
                    dictionary['root'] = d._.dep_arc
                sentence.append(dictionary)
            dict_list.append(sentence)
        return dict_list
    

    
class Pipeline():
    def __new__(cls, pipeline):
        language = NLP('fa', pipeline)
        nlp = language.nlp
        return nlp
    
    
def load_pipline(pipelines):
    language = NLP('fa', pipelines)
    nlp = language.nlp
    return nlp

# if __name__ == '__main__':
# #     pipelines =  'tok,lem,pos,dep'
#     pipelines =  'lem'
# #     lang = 'fa' # delete
    
# #     language = NLP(lang, pipelines)
# #     nlp = language.nlp
#     nlp = load_pipline(pipelines)
    
#     print(nlp.pipe_names)
#     print(nlp.analyze_pipes(pretty=True))

#     doc = nlp('من دیروز به کتابخانه رفتم! فردا به مدرسه می‌روم.')
#     print(doc)
# #     print(language.to_json(doc))
