import string
from pathlib import Path
from dadmatools.utils.patterns import *
import html2text


prefix = str(Path(__file__).parent.absolute()).replace('models', '')
save_dir = '/utils/'


class Normalizer:
    def __init__(
        self,
        full_cleaning=False,
        unify_chars=True,
        refine_punc_spacing=True,
        remove_extra_space=True,
        remove_puncs=False,
        remove_html=False,
        remove_stop_word=False,
        replace_email_with=None,
        replace_number_with=None,
        replace_url_with=None,
        replace_mobile_number_with=None,
        replace_emoji_with=None,
        replace_home_number_with=None
    ):
        self.replace_patterns = []
        self.remove_html = remove_html
        self.remove_puncs = remove_puncs
        self.remove_stop_word = remove_stop_word
        self.STOPWORDS = open(prefix+save_dir+'stopwords-fa.py', encoding='utf-8', errors='ignore').read().splitlines()
        self.PUNCS = string.punctuation.replace('<', '').replace('>', '') + '،؟'
        if full_cleaning:
            self.remove_html = True
            self.remove_puncs = True
            self.remove_stop_word = True
            replace_email_with = ''
            replace_url_with = ''
            replace_emoji_with = ''
            replace_number_with = ''
            replace_mobile_number_with = ''
            replace_home_number_with = ''
        if unify_chars or full_cleaning:
            self.replace_patterns.extend(PERSIAN_CHAR_UNIFY_LIST)
        if replace_email_with is not None or full_cleaning:
            self.replace_patterns.append((EMAIL_REGEX, replace_email_with))

        if replace_url_with is not None or full_cleaning:
            self.replace_patterns.append((URL_REGEX, replace_url_with))
        if replace_mobile_number_with is not None or full_cleaning:
            self.replace_patterns.append((MOBILE_PHONE_REGEX, replace_mobile_number_with))
        if replace_emoji_with is not None or full_cleaning:
            self.replace_patterns.append((EMOJI_REGEX, replace_emoji_with))
        if replace_home_number_with is not None or full_cleaning:
            self.replace_patterns.append((HOME_PHONE_REGEX, replace_home_number_with))
        if replace_number_with is not None or full_cleaning:
            self.replace_patterns.append((NUMBERS_REGEX, replace_number_with))
        if remove_extra_space or full_cleaning:
            self.replace_patterns.extend(REMOVE_SPACE_PATTERNS)
        if refine_punc_spacing or full_cleaning:
            self.replace_patterns.extend(PUNC_SPACING_PATTERNS)

        # compile
        for index, (pattern, repl) in enumerate(self.replace_patterns):
            self.replace_patterns[index] = (re.compile(pattern), repl)
        # save_dir = 'saved_models/normalizer/normalize/'

    def replace_text(self, text):
        if self.remove_html:
            text = html2text.html2text(text)
        for pattern, repl in self.replace_patterns:
            text = pattern.sub(repl, text)
        return text

    def normalize(self, text):
        text = self.replace_text(text)
        if self.remove_puncs:
            for p in self.PUNCS:
                text = text.replace(p, ' ')
            text = ' '.join(text.split())
            # text = text.translate(str.maketrans('d', 'd', self.PUNCS))
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


if __name__ == '__main__':
    text = '<p> karrabi.mohammad@gmail.com  ایمیل ۲۳۳ ۲۳.۳ . <p>'
    # normalizer = Normalizer(remove_html=True)
    normalizer = Normalizer(full_cleaning=True)
    text = normalizer.normalize(text)
    print(text)