import string
from pathlib import Path
from dadmatools.models.normalize.patterns import *

# import dadmatools.pipeline.download as dl


prefix = str(Path(__file__).parent.absolute()).replace('models', '')
# save_dir = 'saved_models/normalizer/normalize/'
save_dir = 'models/normalize/'

class Normalizer:
    def __init__(self, unify_chars=True, refine_punc_spacing=False, remove_extra_space=True,
                 replace_email=False,replace_number=False, replace_url=False,
                 remove_stop_word=False, remove_puncs=False, replace_mobile_number=False,
                 replace_emoji=False, replace_home_number= False):
        self.replace_patterns = []
        self.remove_puncs = remove_puncs
        self.remove_stop_word = remove_stop_word
        self.STOPWORDS = open(prefix+save_dir+'stopwords-fa.py').read().splitlines()
        self.PUNCS = string.punctuation.replace('<', '').replace('>', '') + '،؟'

        if unify_chars:
            self.replace_patterns.extend(PERSIAN_CHAR_UNIFY_LIST)
        if replace_email:
            self.replace_patterns.append((EMAIL_REGEX, '<EMAIL>'))

        if replace_url:
            self.replace_patterns.append((URL_REGEX, '<URL>'))
        if replace_mobile_number:
            self.replace_patterns.append((MOBILE_PHONE_REGEX, '<MOBILE NUMBER>'))
        if replace_emoji:
            self.replace_patterns.append((EMOJI_REGEX, '<EMOJI>'))
        if replace_home_number:
            self.replace_patterns.append((HOME_PHONE_REGEX, '<PHONE NUMBER>'))
        if replace_number:
            self.replace_patterns.append((NUMBERS_REGEX, '<NUMBER>'))
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
        text = self.replace_text(text)
        if self.remove_puncs:
            text = text.translate(str.maketrans('', '', self.PUNCS))
        tokens = text.split()
        if self.remove_stop_word:
            tokens = [tok for tok in tokens if tok not in self.STOPWORDS]
        text = ' '.join(tokens)
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