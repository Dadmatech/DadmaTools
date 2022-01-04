import nltk


class FindChunks():

    def __init__(self):
        self.grammar = r"""
                        VPRT: {<.*,compound:lvc,NOUN>(<و,cc,CCONJ><.*conj,NOUN>)?<.*,PRON>?}
                        VPRT: {<.*,compound:lvc,ADJ>(<و,cc,CCONJ><.*conj,ADJ>)?<.*,PRON>?}
                        VP: {<.*,compound:lvc,NOUN><.*,ADP><.*,NOUN><.*,VERB><.*,AUX>?}
                        VP: {<.*,compound:lvc,NOUN><.*,ADP><.*,NOUN><.*,AUX>?<.*,VERB>}
                        VP: {<.*,ADP><.*,compound:lvc,PRON><.*,VERB><.*,AUX>?}
                        VP: {<.*,ADP><.*,compound:lvc,PRON><.*,AUX>?<.*,VERB>}
                        VP: {<.*,ADP><.*,compound:lvc,NOUN><.*,VERB><.*,AUX>?}
                        VP: {<.*,ADP><.*,compound:lvc,NOUN><.*,AUX>?<.*,VERB>}
                        VP: {<VPRT><.*,VERB><.*,AUX>?}
                        VP: {<VPRT><.*,AUX>?<.*,VERB>}
                        VP: {<.*,AUX>?<.*,VERB>}
                        VP: {<.*,VERB><.*,AUX>*}
                        VP: {<.*,AUX>}
                        ADJP: {<.*advmod,ADV>?<.*,ADJ>(<،,punct,PUNCT><.*,ADV>?<.*conj,ADJ>)+<و,cc,CCONJ><.*,ADV>?<.*conj,ADJ>}
                        ADJP: {<.*advmod,ADV>?<.*,ADJ>(<و,cc,CCONJ><.*,ADV>?<.*conj,ADJ>)*}
                        NUMP: {<.*,NUM>+(<و,cc,CCONJ><.*,NUM>+)+}
                        NUMP: {<.*,NUM>+((<،,punct,PUNCT><.*,NUM>)+<و,cc,CCONJ><.*,NUM>)}
                        PRNP: {<.*nmod,PROPN><.*nmod,PROPN>?}
                        NP: {<.*,INTJ><.*vocative,NOUN><AJP>?}
                        NP: {<.*vocative,NOUN><.*,INTJ>}
                        NP: {<.*,NOUN>(<،,punct,PUNCT><.*conj,NOUN>)+<و,cc,CCONJ><.*conj,NOUN><PRNP>?}
                        NP: {<.*,NOUN>(<و,cc,CCONJ><.*conj,NOUN>)+<PRNP>?}
                        NP: {<.*,DET>+<.*,NOUN><AJP>?<PRNP>?}
                        NP: {<.*,DET>?<NUMP><.*,NOUN><AJP>?<PRNP>?}
                        NP: {<.*,DET>?<.*,NOUN><NUMP><PRNP>?}
                        NP: {<AJP>?<.*,NOUN><AJP>?<PRNP>?}
                        NP: {<.*,PROPN><.*flat:name,PROPN>*(<،,punct,PUNCT><.*conj,PROPN><.*flat:name,PROPN>*)+<و,cc,CCONJ><.*conj,PROPN><.*flat:name,PROPN>*}
                        NP: {<.*,PROPN><.*flat:name,PROPN>*(<و,cc,CCONJ><.*conj,PROPN><.*flat:name,PROPN>*)*}
                        NP: {<.*,PRON><.*nmod,PROPN>?(<،,punct,PUNCT><.*conj,PRON><.*nmod,PROPN>?)+<و,cc,CCONJ><.*conj,PRON><.*nmod,PROPN>?}
                        NP: {<.*,PRON><.*nmod,PROPN>?(<و,cc,CCONJ><.*conj,PRON><.*nmod,PROPN>)*}
                        NP: {<NUMP>}
                        ADVP: {<.*,ADV>}
                        PP: {<.*,ADP>}
                        INTJP: {<.*case,INTJ>?<.*,INTJ>}
                        PARTP: {<.*,PART>}
                        CCONJP: {<.*,CCONJ>}
                        SCONJP: {<.*,SCONJ>}        
                        """

        self.cp = nltk.RegexpParser(self.grammar)

    def convert_nestedtree2rawstring(self, tree, d=0):
        s = ''
        for item in tree:
            if isinstance(item, tuple):
                s += item[0] + ' '
            elif d >= 1:
                news = self.convert_nestedtree2rawstring(item, d + 1)
                s += news + ' '
            else:
                tag = item._label
                news = '[' + self.convert_nestedtree2rawstring(item, d + 1) + ' ' + tag + ']'
                s += news + ' '
        return s.strip()

    def chunk_sentence(self, pos_taged_tuples):
        return self.cp.parse(pos_taged_tuples)



def load_model():
    chunker_dadma = FindChunks()
    return chunker_dadma

def chunk(model, sentence):
    chnk_tags = list(zip([t.text for t in sentence ],[str(d.text+','+ d.dep_ +','+ d.pos_) for d in sentence ]))
    chunk_tree = model.convert_nestedtree2rawstring(model.chunk_sentence(chnk_tags))
    return chunk_tree

