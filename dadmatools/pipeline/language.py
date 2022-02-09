import spacy
from spacy import displacy
from spacy import pipeline
from spacy.language import Language
from spacy.tokens import Doc, Token, Span
from spacy.pipeline import Sentencizer

import dadmatools.models.normalizer as normalizer
import dadmatools.models.tokenizer as tokenizer
import dadmatools.models.mw_tokenizer as mwt
import dadmatools.models.lemmatizer as lemmatizer
import dadmatools.models.postagger as tagger
import dadmatools.models.dependancy_parser as dp
import dadmatools.models.constituency_parser as conspars
import dadmatools.models.ner as ner
import dadmatools.models.chunker as chunker
import dadmatools.models.kasreh as kasreh


class NLP():
    """
    In this class a blank pipeline in created and it is initialized based on our trained models
    possible pipelines: [tokenizer, lemmatize, postagger, dependancyparser]
    """
    tokenizer_model = None
    mwt_model = None
    lemma_model = None
    postagger_model = None
    depparser_model = None
    consparser_model = None
    chunker_model = None
    normalizer_model = None
    ner_model = None
    kasreh_model = None
    
    Token.set_extension("dep_arc", default=None)
    Doc.set_extension("sentences", default=None)
    Doc.set_extension("chunks", default=None)
    Doc.set_extension("constituency", default=None)
    Doc.set_extension("ners", default=None)
    Doc.set_extension("kasreh_ezafe", default=None)
    
    global nlp
    nlp = None
    
    def __init__(self, lang, pipelines):
        
        global nlp
        nlp = spacy.blank(lang)
        self.nlp = nlp
        
        self.dict = {'tok':'tokenizer', 'lem':'lemmatize', 'pos':'postagger', 'dep':'dependancyparser', 'cons':'constituencyparser'}
        self.pipelines = pipelines.split(',')
        
        # if 'def-norm' in pipelines:
        #     global normalizer_model
        #     normalizer_model = normalizer.load_model()
        #     self.nlp.add_pipe('normalizer', first=True)

        global tokenizer_model
        tokenizer_model = tokenizer.load_model()
        self.nlp.add_pipe('tokenizer')
        
        global mwt_model
        mwt_model = mwt.load_model()

        if 'lem' in pipelines:
            global lemma_model
            lemma_model = lemmatizer.load_model()
            self.nlp.add_pipe('lemmatize')
        
        if ('dep' or 'chunk') in pipelines:
            global depparser_model
            depparser_model = dp.load_model()
            self.nlp.add_pipe('dependancyparser')
        
        if ('pos' or 'chunk') in pipelines:
            global postagger_model
            postagger_model = tagger.load_model()
            self.nlp.add_pipe('postagger')

        if 'chunk' in pipelines:
            global chunker_model
            chunker_model = chunker.load_model()
            self.nlp.add_pipe('chunking')
        
        if 'cons' in pipelines:
            global consparser_model
            consparser_model = conspars.load_model()
            self.nlp.add_pipe('constituencyparser')
        
        if 'ner' in pipelines:
            global ner_model
            ner_model = ner.load_model()
            self.nlp.add_pipe('ners')
        
        if 'kasreh' in pipelines:
            global kasreh_model
            kasreh_model = kasreh.load_model()
            self.nlp.add_pipe('kasreh_ezafe')
    
    # @Language.component('normalizer')
    # def tokenizer(doc):
    #     model = normalizer_model
    #     with doc.retokenize() as retokenizer:
    #         retokenizer.merge(doc[0:len(doc)])
    #     norm_text = model.normalize(doc.text)
    #     words = norm_text.split(' ')
    #     spaces = [True for t in words]
    #     doc = Doc(nlp.vocab, words=words, spaces=spaces)
    #     return doc

    @Language.component('tokenizer', retokenizes=True)
    def tokenizer(doc):
        model, args = tokenizer_model
        model_mwt, args_mwt = mwt_model
        
        with doc.retokenize() as retokenizer:
            retokenizer.merge(doc[0:len(doc)])
        starts = []
        tokens_list = tokenizer.tokenizer(model, args, doc.text)
        tokens_list = mwt.mwt(model_mwt, args_mwt, tokens_list)
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
    
    @Language.component('chunking')
    def constituencyparser(doc):
        model = chunker_model

        chunks = []
        for sent in doc._.sentences:
            chu = chunker.chunk(model, sent)
            chunks.append(chu)
        
        doc._.chunks = chunks
        
        return doc

    @Language.component('constituencyparser')
    def constituencyparser(doc):
        model = consparser_model
        
        constitu_parses = []
        for sent in doc._.sentences:
            ## getting the constituency of the sentence
            cons_res = conspars.cons_parser(model, sent.text)
            constitu_parses.append(cons_res)
        
        doc._.constituency = constitu_parses
        
        return doc
    
    @Language.component('ners')
    def namedentity(doc):
        model = ner_model
        
        ners = []
        for sent in doc._.sentences:
            ## getting the IOB tags of the sentence
            ners.append(ner.ner(model, sent.text))
        
        doc._.ners = ners
        
        return doc
    
    @Language.component('kasreh_ezafe')
    def kasrehezafe(doc):
        model = kasreh_model
        
        all_kasreh = []
        for sent in doc._.sentences:
            ## getting the IOB tags of the sentence
            all_kasreh.append(kasreh.kasreh_ezafe(model))
        
        doc._.kasreh_ezafe = all_kasreh
        
        return doc

   
class Pipeline():
    def __new__(cls, pipeline):
        language = NLP('fa', pipeline)
        nlp = language.nlp
        return nlp 

        
def load_pipline(pipelines):
    language = NLP('fa', pipelines)
    nlp = language.nlp
    return nlp

def to_json(pipelines, doc):
    dict_list = []
    for sent in doc._.sentences:
        sentence = []
        for i, d in enumerate(sent):
            dictionary = {}
            dictionary['id'] = i+1
            dictionary['text'] = d.text
            if 'lem' in pipelines: dictionary['lemma'] = d.lemma_
            if 'pos' in pipelines: dictionary['pos'] = d.pos_
            if 'dep' in pipelines: 
                dictionary['rel'] = d.dep_
                dictionary['root'] = d._.dep_arc
            sentence.append(dictionary)
        dict_list.append(sentence)
    return dict_list
 
