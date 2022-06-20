from __future__ import unicode_literals

import nltk
import NERDA
kasre_model = torch.load('kasreh_ezafeh_nerda_2epoch.pt')
class FindChunks():

    def __init__(self):
        self.grammar = r"""
                        PP: {<.*,را,.*,ADP>}
                        VP: {<.*,compound:lvc,NOUN><.*,ADP><.*,NOUN><.*,AUX><.*,VERB>}
                        VP: {<.*,compound:lvc,NOUN><.*,ADP><.*,NOUN><.*,VERB><.*,AUX>*}
                        VP: {<.*,compound:lvc,NOUN><.*,NOUN><.*,AUX><.*,VERB>}
                        VP: {<.*,compound:lvc,NOUN><.*,NOUN><.*,VERB><.*,AUX>*}
                        VP: {<.*,ADP><.*,compound:lvc,PRON><.*,AUX><.*,VERB>}
                        VP: {<.*,ADP><.*,compound:lvc,PRON><.*,VERB><.*,AUX>*}
                        VP: {<.*,ADP><.*,compound:lvc,NOUN><.*,AUX><.*,VERB>}
                        VP: {<.*,ADP><.*,compound:lvc,NOUN><.*,VERB><.*,AUX>*}
                        VP: {<.*,compound:lvc,NOUN>(<.*,و,.*,CCONJ><.*conj,NOUN>)?<.*,PRON>?<.*,AUX><.*,VERB>}
                        VP: {<.*,compound:lvc,NOUN>(<.*,و,.*,CCONJ><.*conj,NOUN>)?<.*,PRON>?<.*,VERB><.*,AUX>*}
                        VP: {<.*,compound:lvc,ADJ>(<.*,و,.*,CCONJ><.*conj,ADJ>)?<.*,PRON>?<.*,AUX><.*,VERB>}
                        VP: {<.*,compound:lvc,ADJ>(<.*,و,.*,CCONJ><.*conj,ADJ>)?<.*,PRON>?<.*,VERB><.*,AUX>*}
                        VP: {<.*,AUX><.*,VERB>}
                        VP: {<.*,VERB><.*,AUX>*}
                        VP: {<.*,AUX>}

                        AJPEZ: {<.*,ADP><KASREH,.*amod,NOUN>}
                        AJPEZ: {<.*,NUM><KASREH,.*amod,NOUN>}
                        AJPEZ: {(<.*advmod,ADV>?<KASREH,.*amod,ADJ>+)<.*,ADV>?<KASREH,.*amod,ADJ>}
                        AJPEZ: {<.*advmod,ADV>?<.*amod,ADJ>(<.*,،,punct,PUNCT><.*,ADV>?<.*conj,ADJ>)+<.*,و,.*,CCONJ><.*,ADV>?<KASREH,.*conj,ADJ>}
                        AJPEZ: {<.*advmod,ADV>?<.*amod,ADJ>(<.*,و,.*,CCONJ><.*,ADV>?<.*conj,ADJ>)*<.*,و,.*,CCONJ><.*,ADV>?<KASREH,.*conj,ADJ>}
                        AJPEZ: {<.*advmod,ADV>?<KASREH,.*amod,ADJ>}

                        NUMPEZ: {<.*,NUM>+(<.*,و,.*,CCONJ><.*,NUM>+)*<.*,و,.*,CCONJ><.*,NUM>?<KASREH,.*,NUM>}
                        NUMPEZ: {<.*,NUM>+(<.*,،,punct,PUNCT><.*,NUM>)+<.*,و,.*,CCONJ><.*,NUM>?<KASREH,.*,NUM>}
                        NUMPEZ: {<.*,NUM><KASREH,.*,NUM>}
                        NUMPEZ: {<KASREH,.*,NUM>}

                        AJP: {<.*,ADP><.*,ADP>(<.*amod,PRON>|<.*amod,NOUN>)}
                        AJP: {<.*,ADP><.*amod,NOUN>}
                        AJP: {<.*,NUM><.*amod,NOUN>}
                        AJP: {(<.*advmod,ADV>?<KASREH,.*amod,ADJ>+)<.*,ADV>?<N-KASREH,.*amod,ADJ>}
                        AJP: {<.*advmod,ADV>?<.*amod,ADJ>(<.*,،,punct,PUNCT><.*,ADV>?<.*conj,ADJ>)+<.*,و,.*,CCONJ><.*,ADV>?<N-KASREH,.*conj,ADJ>}
                        AJP: {<.*advmod,ADV>?<N-KASREH,.*amod,ADJ>(<.*,و,.*,CCONJ><.*,ADV>?<N-KASREH,.*conj,ADJ>)*}

                        NUMP: {<.*,NUM>+(<.*,و,.*,CCONJ><N-KASREH,.*,NUM>+)+}
                        NUMP: {<.*,NUM>+(<.*,و,.*,CCONJ><N-KASREH,.*,NUM>+)*<.*,و,.*,CCONJ><N-KASREH,.*,NOUN>}
                        NUMP: {<N-KASREH,.*,NUM>+((<.*,،,punct,PUNCT><.*,NUM>+)+<.*,و,.*,CCONJ><N-KASREH,.*,NUM>)?}

                        INTJP: {<.*,INTJ><KASREH,.*vocative,NOUN><AJP>(<.*nmod,PRON><.*nmod,PRON>?)?}
                        INTJP: {<.*,INTJ><KASREH,.*vocative,NOUN><AJPEZ>?}
                        INTJP: {<.*,INTJ><KASREH,.*vocative,NOUN>(<.*nmod,PRON><.*nmod,PRON>?)?}
                        INTJP: {<.*vocative,NOUN><.*,INTJ>}
                        INTJP: {<.*case,INTJ><.*,INTJ>}

                        NP: {<.*,NOUN><.*,flat,NOUN>+<.*amod,ADJ>?}
                        NP: {<.*,DET>+<AJP>?<KASREH,.*,NOUN><AJP>(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,DET>?<NUMP><KASREH,.*,NOUN><AJP>(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,DET>?<KASREH,.*,NOUN><NUMP>(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<AJP>?<KASREH,.*,NOUN><AJP>(<.*nmod,PRON><.*nmod,PRON>?)?}

                        NPEZ: {<.*advmod,ADJ><.*,compound:lvc,NOUN><.*,ADP><.*,NOUN>(<KASREH,.*تن,.*,NOUN>|<KASREH,.*دن,.*,NOUN>)<AJPEZ>?}
                        NPEZ: {<.*advmod,ADJ><.*,compound:lvc,NOUN><.*,NOUN>(<KASREH,.*تن,.*,NOUN>|<KASREH,.*دن,.*,NOUN>)<AJPEZ>?}
                        NPEZ: {<.*advmod,ADJ><.*,ADP><.*,compound:lvc,PRON>(<KASREH,.*تن,.*,NOUN>|<KASREH,.*دن,.*,NOUN>)<AJPEZ>?}
                        NPEZ: {<.*advmod,ADJ><.*,ADP><.*,compound:lvc,NOUN>(<KASREH,.*تن,.*,NOUN>|<KASREH,.*دن,.*,NOUN>)<AJPEZ>?}
                        NPEZ: {<.*advmod,ADJ><.*,compound:lvc,NOUN>(<.*,و,.*,CCONJ><.*conj,NOUN>)?(<KASREH,.*تن,.*,NOUN>|<KASREH,.*دن,.*,NOUN>)<AJPEZ>?}
                        NPEZ: {<.*advmod,ADJ><.*,compound:lvc,ADJ>(<.*,و,.*,CCONJ><.*conj,ADJ>)?(<KASREH,.*تن,.*,NOUN>|<KASREH,.*دن,.*,NOUN>)<AJPEZ>?}
                        NPEZ: {<.*advmod,ADJ>?(<KASREH,.*تن,.*,NOUN>|<KASREH,.*دن,.*,NOUN>)<AJPEZ>?}
                        NPEZ: {<.*,DET>+<AJP>?<KASREH,.*,NOUN><AJPEZ>?}
                        NPEZ: {<.*,DET>?<NUMP><KASREH,.*,NOUN><AJPEZ>?}
                        NPEZ: {<.*,DET>?<KASREH,.*,NOUN><NUMPEZ>}
                        NPEZ: {<AJP>?<KASREH,.*,NOUN><AJPEZ>?}

                        NP: {<.*advmod,ADJ>?<.*,compound:lvc,NOUN><.*,ADP><.*,NOUN>(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)<AJP>?(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*advmod,ADJ>?<.*,compound:lvc,NOUN><.*,NOUN>(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)<AJP>?(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*advmod,ADJ>?<.*,ADP><.*,compound:lvc,PRON>(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)<AJP>?(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*advmod,ADJ>?<.*,ADP><.*,compound:lvc,NOUN>(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)<AJP>?(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*advmod,ADJ>?<.*,compound:lvc,NOUN>(<.*,و,.*,CCONJ><.*conj,NOUN>)?(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)<AJP>?(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*advmod,ADJ>?<.*,compound:lvc,ADJ>(<.*,و,.*,CCONJ><.*conj,ADJ>)?(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)<AJP>?(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*advmod,ADJ>?(<.*تن,.*,NOUN>|<.*دن,.*,NOUN>)<AJP>?(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,DET>+<AJP>?<.*,NOUN>(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,DET>?<NUMP><.*,NOUN>(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<AJP>?<.*,NOUN><AJP>?(<.*nmod,PRON><.*nmod,PRON>?)?}
                        NP: {<.*,PRON><.*nmod,PRON>?(<.*,،,punct,PUNCT><.*conj,PRON><.*nmod,PRON>?)+<.*,و,.*,CCONJ><.*conj,PRON><.*nmod,PRON>?}
                        NP: {<.*,PRON><.*nmod,PRON>?(<.*,و,.*,CCONJ><.*conj,PRON><.*nmod,PRON>?)*}

                        MNP: {<NPEZ>+<NP>}
                        MNP: {<NP>}
                        CONP: {<MNP>(<.*,،,punct,PUNCT><MNP>)+<.*,و,.*,CCONJ><MNP>}
                        CONP: {<MNP>(<.*,و,.*,CCONJ><MNP>)+}

                        ADJP: {<.*advmod,ADV>?<.*ADJ>(<.*,،,punct,PUNCT><.*ADV>?<.*conj,ADJ>)+<.*,و,.*,CCONJ><.*ADV>?<.*conj,ADJ>}
                        ADJP: {<.*advmod,ADV>?<.*ADJ>(<.*,و,.*,CCONJ><.*ADV>?<.*conj,ADJ>)*}
                        ADVP: {<.*,ADV>}
                        PP: {<.*,ADP>}
                        PARTP: {<.*,PART>}
                        ADJP: {<AJP>|<AJPEZ>}
                        NP: {<NUMP>|<NUMPEZ>|<NPEZ>|<MNP>|<CONP>}

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

  resp = kasre_model.predict_text(sent_t, sent_tokenize=sent_tokenize, word_tokenize=word_tokenize)
  sentences = resp[0]
  tags = resp[1]
  sent_tags = zip(sentences, tags)
  for sent, tags in sent_tags:
      for token, t in zip(sent,tags):
          kasreh_tags.append(t)

def load_model():
    chunker_dadma = FindChunks()
    return chunker_dadma

def chunk(kasre_model,model, sentence):
    for s in sentence:
        resp = kasre_model.predict_text(d, sent_tokenize=sent_tokenize, word_tokenize=return(d.tok_))
        sent_tags = zip(resp[0], resp[1])
        for sent, tags in sent_tags:
        for token, t in zip(sent,tags):
        kasreh_tags.append(t)
    chnk_tags = list(zip([t.text for t in sentence ],[str(k+','+ d.text+','+ d.dep_ +','+ d.pos_) for d,k in zip(sentence,kasreh_tags) ]))
    chunk_tree = model.convert_nestedtree2rawstring(model.chunk_sentence(chnk_tags))
    return chunk_tree

