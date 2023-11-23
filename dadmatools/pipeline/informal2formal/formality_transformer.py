
import pickle
from .kenlm_wrapper import Kelm_Wrapper
from .OneShotTransformer import OneShotTransformer
from .VerbHandler import VerbHandler
import kenlm
from .tokenizer import InformalTokenizer


class FormalityTransformer:
    def __init__(self, asset_file_addr, verbs_csv_addr, irregular_verbs_mapper_addr, lm_addr ):
        assets = pickle.load(open(asset_file_addr, 'rb'))
        self.vocab = assets['vocab']
        self.word_ends_tanvin = assets['word_ends_tanvin']
        self.non_hidden_h_words = assets['non_hidden_h_words']
        self.isolated_words = assets['isolated_words']
        self.ignore_words = assets['ignore_words']
        self.mapper = assets['mapper']
        self.postfix_mapper = assets['postfix_mapper']
        postfixes = assets['postfixes']

        self.informal_tokenizer = InformalTokenizer(self.vocab, postfixes)
        self.verb_handler = VerbHandler(csv_verb_addr=verbs_csv_addr, csv_irregular_verbs_mapper=irregular_verbs_mapper_addr)
        self.oneshot_transformer = OneShotTransformer(self.vocab, self.mapper, self.verb_handler.informal_to_formal,
                                                      ignore_words=self.ignore_words,
                                                      postfix_mapper=self.postfix_mapper,
                                                      isolated_words=self.isolated_words,
                                                      non_hidden_h_words=self.non_hidden_h_words)
        lm_model = kenlm.Model(lm_addr)
        self.lm_obj = Kelm_Wrapper(lm_model)


    def should_filtered_by_one_bigram(self, lemma, word, original_word):
        NIM_FASELE = '‌'
        return original_word in self.vocab and (len(word.split()) > 1 or NIM_FASELE in word)

    def repalce_for_gpt2(self, word_repr):
        if word_repr in self.word_ends_tanvin:
            return word_repr[:-2] + 'ا'
        return word_repr
