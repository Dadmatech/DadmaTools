import nltk


class FindChunks():

    def __init__(self):
        self.grammar = r"""
                        PP: {<را,.*,ADP>}
                        VP: {<.*,compound:lvc,NOUN><.*,ADP><.*,NOUN><.*,VERB><.*,AUX>?}
                        VP: {<.*,compound:lvc,NOUN><.*,ADP><.*,NOUN><.*,AUX>?<.*,VERB>}
                        VP: {<.*,compound:lvc,NOUN><.*,NOUN><.*,VERB><.*,AUX>?}
                        VP: {<.*,compound:lvc,NOUN><.*,NOUN><.*,AUX>?<.*,VERB>}
                        VP: {<.*,ADP><.*,compound:lvc,PRON><.*,VERB><.*,AUX>?}
                        VP: {<.*,ADP><.*,compound:lvc,PRON><.*,AUX>?<.*,VERB>}
                        VP: {<.*,ADP><.*,compound:lvc,NOUN><.*,VERB><.*,AUX>?}
                        VP: {<.*,ADP><.*,compound:lvc,NOUN><.*,AUX>?<.*,VERB>}
                        VP: {<.*,compound:lvc,NOUN>(<و,.*,CCONJ><.*conj,NOUN>)?<.*,PRON>?<.*,VERB><.*,AUX>?}
                        VP: {<.*,compound:lvc,NOUN>(<و,.*,CCONJ><.*conj,NOUN>)?<.*,PRON>?<.*,AUX>?<.*,VERB>}
                        VP: {<.*,compound:lvc,ADJ>(<و,.*,CCONJ><.*conj,ADJ>)?<.*,PRON>?<.*,VERB><.*,AUX>?}
                        VP: {<.*,compound:lvc,ADJ>(<و,.*,CCONJ><.*conj,ADJ>)?<.*,PRON>?<.*,AUX>?<.*,VERB>}
                        VP: {<.*,AUX><.*,VERB>}
                        VP: {<.*,VERB><.*,AUX>*}
                        VP: {<.*,AUX>}
                        AJP: {<.*advmod,ADV>?<.*amod,ADJ>(<،,punct,PUNCT><.*,ADV>?<.*conj,ADJ>)+<و,.*,CCONJ><.*,ADV>?<.*conj,ADJ>}
                        AJP: {<.*advmod,ADV>?<.*amod,ADJ>(<و,.*,CCONJ><.*,ADV>?<.*conj,ADJ>)*}
                        NUMP: {<.*,NUM>+(<و,.*,CCONJ><.*flat:num,NUM>+)+}
                        NUMP: {<.*,NUM>+(<و,.*,CCONJ><.*flat:num,NOUN>+)+}
                        NUMP: {<.*,NUM>+((<،,punct,PUNCT><.*,NUM>)+<و,.*,CCONJ><.*flat:num,NUM>)?}
                        NUMP: {<.*,NUM>+((<،,punct,PUNCT><.*,NUM>)+<و,.*,CCONJ><.*flat:num,NOUN>)?}
                        NP: {<.*,INTJ><.*vocative,NOUN><AJP>?}
                        NP: {<.*vocative,NOUN><.*,INTJ>}
                        NP: {<.*,compound:lvc,NOUN><.*,ADP><.*,NOUN>(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,compound:lvc,NOUN><.*,NOUN>(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,ADP><.*,compound:lvc,PRON>(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,ADP><.*,compound:lvc,NOUN>(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,compound:lvc,NOUN>(<و,.*,CCONJ><.*conj,NOUN>)?(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,compound:lvc,ADJ>(<و,.*,CCONJ><.*conj,ADJ>)?(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,NOUN>(<،,punct,PUNCT><.*conj,NOUN>)+<و,.*,CCONJ><.*conj,NOUN>(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,NOUN>(<و,.*,CCONJ><.*conj,NOUN>)+(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,DET>+<AJP>?<.*,NOUN><AJP>?(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,DET>?<NUMP><.*,NOUN><AJP>?(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,DET>?<.*,NOUN><NUMP>(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<AJP>?<.*,NOUN><AJP>?(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,PROPN><.*flat:name,PROPN>*(<،,punct,PUNCT><.*conj,PROPN><.*flat:name,PROPN>*)+<و,.*,CCONJ><.*conj,PROPN><.*flat:name,PROPN>*}
                        NP: {<.*,PROPN><.*flat:name,PROPN>*(<و,.*,CCONJ><.*conj,PROPN><.*flat:name,PROPN>*)*}
                        NP: {<.*,PRON><.*nmod,PRON>?(<،,punct,PUNCT><.*conj,PRON><.*nmod,PRON>?)+<و,.*,CCONJ><.*conj,PRON><.*nmod,PRON>?}
                        NP: {<.*,PRON><.*nmod,PRON>?(<و,.*,CCONJ><.*conj,PRON><.*nmod,PRON>?)*}
                        ADJP: {<.*advmod,ADV>?<.*ADJ>(<،,punct,PUNCT><.*ADV>?<.*conj,ADJ>)+<و,.*,CCONJ><.*ADV>?<.*conj,ADJ>}
                        ADJP: {<.*advmod,ADV>?<.*ADJ>(<و,.*,CCONJ><.*ADV>?<.*conj,ADJ>)*}
                        ADVP: {<.*,ADV>}
                        PP: {<.*,ADP>}
                        INTJP: {<.*case,INTJ>?<.*,INTJ>}
                        PARTP: {<.*,PART>}  
                        ADJP: {<AJP>}
                        NP: {<NUMP>}
                        NP: {<PRNP>}        
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

