import itertools
import os
from pathlib import Path
import yaml
from .download_utils import download_dataset
import dadmatools.pipeline.informal2formal.utils as utils
from .formality_transformer import FormalityTransformer
from dadmatools.pipeline.persian_tokenization.tokenizer import SentenceTokenizer

FILE_URLS = [
    'https://huggingface.co/datasets/Dadmatech/informal2formal/resolve/main/3gram.bin',
    'https://huggingface.co/datasets/Dadmatech/informal2formal/resolve/main/assets.pkl',
    'https://huggingface.co/datasets/Dadmatech/informal2formal/raw/main/irregular_verb_mapper.csv',
    'https://huggingface.co/datasets/Dadmatech/informal2formal/raw/main/verbs.csv'
]

def translate_short_sent(model, sent):
    out_dict = {}
    txt = utils.cleanify(sent)
    is_valid = lambda w: model.oneshot_transformer.transform(w, None)
    cnd_tokens = model.informal_tokenizer.tokenize(txt, is_valid)
    for tokens in cnd_tokens:
        tokens = [t for t in tokens if t != '']
        new_tokens = []
        for t in tokens:
            new_tokens.extend(t.split())
        txt = ' '.join(new_tokens)
        tokens = txt.split()
        candidates = []
        for index in range(len(tokens)):
            tok = tokens[index]
            cnd = set()
            pos = None
            if model.verb_handler.informal_to_formal(tok):
                pos = 'VERB'
            f_words_lemma = model.oneshot_transformer.transform(tok, pos)
            f_words_lemma = list(f_words_lemma)
            for index, (word, lemma) in enumerate(f_words_lemma):
                if pos != 'VERB' and tok not in model.mapper and model.should_filtered_by_one_bigram(lemma, word, tok):
                    f_words_lemma[index] = (tok, tok)
                else:
                    word_toks = word.split()
                    word_repr = ''
                    for t in word_toks:
                        word_repr += ' ' + t
                    word_repr = word_repr.strip()
                    word_repr = model.repalce_for_gpt2(word_repr)
                    f_words_lemma[index] = (word, word_repr)
            if f_words_lemma:
                cnd.update(f_words_lemma)
            else:
                cnd = {(tok, tok)}
            candidates.append(cnd)
        all_combinations = itertools.product(*candidates)
        all_combinations_list = list(all_combinations)
        for id, cnd in enumerate(all_combinations_list):
            normal_seq = ' '.join([c[0] for c in cnd])
            lemma_seq = ' '.join([c[1] for c in cnd])
            lemma_seq = utils.clean_text_for_lm(lemma_seq)
            out_dict[id] = (normal_seq, lemma_seq)
        candidates = [[item[0] for item in candidate_phrases] for candidate_phrases in candidates]
        return model.lm_obj.get_best(candidates)


def translate(model, sentence_tokenizer, txt):
    sents = sentence_tokenizer.tokenize(txt)
    formal_output = ''
    for sentence in sents:
        formal_sentence = translate_short_sent(model, sentence)
        formal_output += ' ' + formal_sentence
    return formal_output


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


class Informal2Formal:
    def __init__(self, cache_dir: str = 'cache') -> None:
        # download or load files

        download_dataset(FILE_URLS, cache_dir, filename=None)

        # set assets files address
        verbs_csv_addr = os.path.join(cache_dir, 'verbs.csv')
        irregular_verbs_mapper = os.path.join(cache_dir, 'irregular_verb_mapper.csv')
        lm_addr = os.path.join(cache_dir, '3gram.bin')
        assets_file_addr = os.path.join(cache_dir, 'assets.pkl')
        self.sentence_tokenizer = SentenceTokenizer('cache/dadmatools')
        self.model = FormalityTransformer(asset_file_addr=assets_file_addr,
                                          irregular_verbs_mapper_addr=irregular_verbs_mapper,
                                          verbs_csv_addr=verbs_csv_addr, lm_addr=lm_addr)

    def translate(self, txt):
        return translate(self.model, self.sentence_tokenizer, txt)
