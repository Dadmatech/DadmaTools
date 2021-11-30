import pickle
import re
from pathlib import Path
from dadmatools.models.normalize.patterns import PERSIAN_CHAR_UNIFY_LIST, EMAIL_REGEX, NUMBERS_REGEX, URL_REGEX, REMOVE_SPACE_PATTERNS, PUNC_SPACING_PATTERNS

import dadmatools.pipeline.download as dl


prefix = str(Path(__file__).parent.absolute()).replace('models', '')
save_dir = 'saved_models/normalizer/normalize/'

class Normalizer:
    def __init__(self, unify_chars=True, nim_fasele_correction=True, replace_email=True,
                 replace_number=True, replace_url=True,
                 remove_stop_word=False, remove_puncs=False, remove_extra_space=True, refine_punc_spacing=False):
        dl.download_model('normalizer', process_func=dl._unzip_process_func)
        
        self.replace_patterns = []
        self.remove_puncs = remove_puncs
        self.remove_stop_word = remove_stop_word
        self.nim_fasele_correction = nim_fasele_correction
        self.STOPWORDS = open(prefix+save_dir+'stopwords.txt').read().splitlines()
        self.PUNCS = open(prefix+save_dir+'puncs.txt').read().splitlines()

        if nim_fasele_correction:
            self.nim_fasele_mapper = pickle.load(open(prefix+save_dir+'nim_fasele_pairs.pkl', 'rb'))
            self.nim_fasele_mapper = {word1:word2 for word1,word2 in self.nim_fasele_mapper}

        if unify_chars:
            self.replace_patterns.extend(PERSIAN_CHAR_UNIFY_LIST)
        if replace_email:
            self.replace_patterns.append((EMAIL_REGEX, '<EMAIL>'))
        if replace_number:
            self.replace_patterns.append((NUMBERS_REGEX, '<NUMBER>'))
        if replace_url:
            self.replace_patterns.append((URL_REGEX, '<URL>'))

        if remove_extra_space:
            self.replace_patterns.extend(REMOVE_SPACE_PATTERNS)
        if refine_punc_spacing:
            self.replace_patterns.extend(PUNC_SPACING_PATTERNS)

        #compile
        for index, (pattern, repl) in enumerate(self.replace_patterns):
            self.replace_patterns[index] = (re.compile(pattern), repl)

    def replace_text(self, text):
        for pattern, repl in self.replace_patterns:
            text = pattern.sub(repl, text)
        return text


    def normalize(self, text):
        tokens = text.split()
        if self.remove_puncs:
            tokens = [tok for tok in tokens if tok not in self.PUNCS]
        if self.remove_stop_word:
            tokens = [tok for tok in tokens if tok not in self.STOPWORDS]
        if self.nim_fasele_correction:
            tokens = [self.nim_fasele_mapper[tok] if tok in self.nim_fasele_mapper else tok  for tok in tokens ]
        text = ' '.join(tokens)
        text = self.replace_text(text)
        return text


def load_model():
    normalizer = Normalizer()
    return normalizer

def normalizer(model, text):
    text = model.normalize(text)
    return text

# if __name__ == '__main__':
#     text = 'karrabi.mohammad@gmail.com حيواني ایمیل ۲۳۳ ۲۳.۳ .'
#     normalizer = Normalizer()
#     text = normalizer.normalize(text)
#     print(text)