"""
"""

import re
import sys
import unicodedata


# using clean-text source code : https://github.com/jfilter/clean-text/blob/master/cleantext/constants.py
# taken hostname, domainname, tld from URL regex below
EMAIL_REGEX = re.compile(
    r"(?:^|(?<=[^\w@.)]))([\w+-](\.(?!\.))?)*?[\w+-](@|[(<{\[]at[)>}\]])(?:(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)(?:\.(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)*(?:\.(?:[a-z\\u00a1-\\uffff]{2,}))",
    flags=re.IGNORECASE | re.UNICODE,
)

# for more information: https://github.com/jfilter/clean-text/issues/10
# PHONE_REGEX = re.compile(
#     r"((?:^|(?<=[^\w)]))(((\+?[01])|(\+\d{2}))[ .-]?)?(\(?\d{3,4}\)?/?[ .-]?)?(\d{3}[ .-]?\d{4})(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W)))|\+?\d{4,5}[ .-/]\d{6,9}"
# )
HOME_PHONE_REGEX = re.compile(r"(\d{8})|(0\d{2}[-]?\d{8})")
MOBILE_PHONE_REGEX = re.compile(r"((098|\+98)?(0)?9\d{9})")
EMOJI_REGEX = re.compile(pattern="["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    "]+", flags=re.UNICODE)

NUMBERS_REGEX = re.compile(
    r"(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))"
)

LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+")
TWO_LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+((\r\n)|[\n\v])+")
MULTI_WHITESPACE_TO_ONE_REGEX = re.compile(r"\s+")
NONBREAKING_SPACE_REGEX = re.compile(r"(?!\n)\s+")

# source: https://gist.github.com/dperini/729294
# @jfilter: I guess it was changed
URL_REGEX = re.compile(
    r"(?:^|(?<![\w\/\.]))"
    # protocol identifier
    # r"(?:(?:https?|ftp)://)"  <-- alt?
    r"(?:(?:https?:\/\/|ftp:\/\/|www\d{0,3}\.))"
    # user:pass authentication
    r"(?:\S+(?::\S*)?@)?" r"(?:"
    # IP address exclusion
    # private & local networks
    r"(?!(?:10|127)(?:\.\d{1,3}){3})"
    r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    # IP address dotted notation octets
    # excludes loopback network 0.0.0.0
    # excludes reserved space >= 224.0.0.0
    # excludes network & broadcast addresses
    # (first & last IP address of each class)
    r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    r"|"
    # host name
    r"(?:(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)"
    # domain name
    r"(?:\.(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)*"
    # TLD identifier
    r"(?:\.(?:[a-z\\u00a1-\\uffff]{2,}))" r"|" r"(?:(localhost))" r")"
    # port number
    r"(?::\d{2,5})?"
    # resource path
    r"(?:\/[^\)\]\}\s]*)?",
    # r"(?:$|(?![\w?!+&\/\)]))",
    # @jfilter: I removed the line above from the regex because I don't understand what it is used for, maybe it was useful?
    # But I made sure that it does not include ), ] and } in the URL.
    flags=re.UNICODE | re.IGNORECASE,
)


strange_double_quotes = [
    "«",
    "‹",
    "»",
    "›",
    "„",
    "“",
    "‟",
    "”",
    "❝",
    "❞",
    "❮",
    "❯",
    "〝",
    "〞",
    "〟",
    "＂",
]
strange_single_quotes = ["‘", "‛", "’", "❛", "❜", "`", "´", "‘", "’"]

DOUBLE_QUOTE_REGEX = re.compile("|".join(strange_double_quotes))
SINGLE_QUOTE_REGEX = re.compile("|".join(strange_single_quotes))

#using Parsivar Normalizer : https://github.com/ICTRC/Parsivar/blob/master/parsivar/normalizer.py
PERSIAN_CHAR_UNIFY_LIST = [
        ( "ء", "ئ"),
        (r"ٲ|ٱ|إ|ﺍ|أ", r"ا"),
        (r"ﺁ|آ", r"آ"),
        (r"ﺐ|ﺏ|ﺑ", r"ب"),
        (r"ﭖ|ﭗ|ﭙ|ﺒ|ﭘ", r"پ"),
        (r"ﭡ|ٺ|ٹ|ﭞ|ٿ|ټ|ﺕ|ﺗ|ﺖ|ﺘ", r"ت"),
        (r"ﺙ|ﺛ", r"ث"),
        (r"ﺝ|ڃ|ﺠ|ﺟ", r"ج"),
        (r"ڃ|ﭽ|ﭼ", r"چ"),
        (r"ﺢ|ﺤ|څ|ځ|ﺣ", r"ح"),
        (r"ﺥ|ﺦ|ﺨ|ﺧ", r"خ"),
        (r"ڏ|ډ|ﺪ|ﺩ", r"د"),
        (r"ﺫ|ﺬ|ﻧ", r"ذ"),
        (r"ڙ|ڗ|ڒ|ڑ|ڕ|ﺭ|ﺮ", r"ر"),
        (r"ﺰ|ﺯ", r"ز"),
        (r"ﮊ", r"ژ"),
        (r"ݭ|ݜ|ﺱ|ﺲ|ښ|ﺴ|ﺳ", r"س"),
        (r"ﺵ|ﺶ|ﺸ|ﺷ", r"ش"),
        (r"ﺺ|ﺼ|ﺻ", r"ص"),
        (r"ﺽ|ﺾ|ﺿ|ﻀ", r"ض"),
        (r"ﻁ|ﻂ|ﻃ|ﻄ", r"ط"),
        (r"ﻆ|ﻇ|ﻈ", r"ظ"),
        (r"ڠ|ﻉ|ﻊ|ﻋ", r"ع"),
        (r"ﻎ|ۼ|ﻍ|ﻐ|ﻏ", r"غ"),
        (r"ﻒ|ﻑ|ﻔ|ﻓ", r"ف"),
        (r"ﻕ|ڤ|ﻖ|ﻗ", r"ق"),
        (r"ڭ|ﻚ|ﮎ|ﻜ|ﮏ|ګ|ﻛ|ﮑ|ﮐ|ڪ|ك", r"ک"),
        (r"ﮚ|ﮒ|ﮓ|ﮕ|ﮔ", r"گ"),
        (r"ﻝ|ﻞ|ﻠ|ڵ", r"ل"),
        (r"ﻡ|ﻤ|ﻢ|ﻣ", r"م"),
        (r"ڼ|ﻦ|ﻥ|ﻨ", r"ن"),
        (r"ވ|ﯙ|ۈ|ۋ|ﺆ|ۊ|ۇ|ۏ|ۅ|ۉ|ﻭ|ﻮ|ؤ", r"و"),
        (r"ﺔ|ﻬ|ھ|ﻩ|ﻫ|ﻪ|ۀ|ە|ة|ہ", r"ه"),
        (r"ﭛ|ﻯ|ۍ|ﻰ|ﻱ|ﻲ|ں|ﻳ|ﻴ|ﯼ|ې|ﯽ|ﯾ|ﯿ|ێ|ے|ى|ي", r"ی"),
        (r'¬', r'‌'),
        (r'•|·|●|·|・|∙|｡|ⴰ', r'.'),
        (r',|٬|٫|‚|，', r'،'),
        (r'ʕ', r'؟'),
        (r'۰|٠', r'0'),
        (r'۱|١', r'1'),
        (r'۲|٢', r'2'),
        (r'۳|٣', r'3'),
        (r'۴|٤', r'4'),
        (r'۵', r'5'),
        (r'۶|٦', r'6'),
        (r'۷|٧', r'7'),
        (r'۸|٨', r'8'),
        (r'۹|٩', r'9')
        # (r'ـ|ِ|ُ|َ|ٍ|ٌ|ً|', r''),
        # (r'( )+', r' '),
        # (r'(\n)+',  r'\n')
        ]

#using hazm normalizer source code : https://github.com/sobhe/hazm/blob/master/hazm/Normalizer.py
punc_after, punc_before = r'\.:!،؛؟»\]\)\}', r'«\[\(\{'
PUNC_SPACING_PATTERNS = [
				('" ([^\n"]+) "', r'"\1"'),  # remove space before and after quotation
				(' (['+ punc_after +'])', r'\1'),  # remove space before
				('(['+ punc_before +']) ', r'\1'),  # remove space after
				('(['+ punc_after[:3] +'])([^ '+ punc_after +'\d۰۱۲۳۴۵۶۷۸۹])', r'\1 \2'),  # put space after . and :
				('(['+ punc_after[3:] +'])([^ '+ punc_after +'])', r'\1 \2'),  # put space after
				('([^ '+ punc_before +'])(['+ punc_before +'])', r'\1 \2'),  # put space before
			]
REMOVE_SPACE_PATTERNS = [
                            (r'( )+', r' '),
                            (r'(\n)+',  r'\n')
                        ]

PUNCS_REGEX = r'\.:!،؛؟»\]\)\}«\[\(\{'

