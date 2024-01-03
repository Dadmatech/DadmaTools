from functools import reduce
import itertools
import json
import re
import string
import pandas as pd
from dadmatools.pipeline.persian_tokenization.tokenizer import WordTokenizer
from dadmatools.normalizer import Normalizer

normalizer = Normalizer()
tokenizer = WordTokenizer('cache/dadmatools')
# tokenizer = WordTokenizer(separate_emoji=True)


def seprate_emoji_string(txt):
        try:
            oRes = re.compile(u'(['
                              u'\U0001F300-\U0001F64F'
                              u'\U0001F680-\U0001F6FF'
                              u'\u2600-\u26FF\u2700-\u27BF]+)',
                              re.UNICODE)
        except re.error:
            oRes = re.compile(u'(('
                              u'\ud83c[\udf00-\udfff]|'
                              u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
                              u'[\u2600-\u26FF\u2700-\u27BF])+)',
                              re.UNICODE)

        return oRes.sub(r'  \1  ', txt)

def cleanify(txt):
    txt = txt.strip()
    txt = re.sub('\s+', ' ', txt)
    txt = re.sub('\u200f', '', txt)
    txt = re.sub('‌+', '‌', txt)
    txt = re.sub('‌ ', ' ', txt)
    txt = re.sub(' ‌', ' ', txt)
    txt = normalizer.normalize(txt)
    txt = seprate_emoji_string(txt)
    txt = ' '.join([word for sent in tokenizer.tokenize(txt) for word in sent])
    return txt




def clean_text_for_lm(txt):
    ignore_chars = '.1234567890!@#$%^&*()_+۱۲۳۴۵۶۷۸۹÷؟×−+?><}،,{":' + string.ascii_lowercase + string.ascii_uppercase
    tokens = txt.split()
    clean_tokens = [t for t in tokens if not (any(ic in t for ic in ignore_chars) or if_emoji(t))]
    return ' '.join(clean_tokens)


def add_to_mapper(mapping_list):
    print(len(mapping_list))
    df = pd.read_csv('resources/mapper.csv', delimiter=',', index_col=None)
    print(df.columns)
    for item in mapping_list:
        df = df.append({'formal': item[1], 'informal': item[0]}, ignore_index=True)
    df.to_csv('resources/mapper.csv', index=False)


def extract_non_convertable_words(corpus_addr, tokenizer, normalizer, transformer, output_addr, vocab):
    f = open(corpus_addr)
    non_convertables = {}
    seen_words = set()
    nim_fasele = '‌'
    for i, line in enumerate(f):
        print(i)
        # if i > 500:
        #     break
        line = normalizer.normalize(line)
        tokens = tokenizer.tokenize(line)
        for t in tokens:
        #     if nim_fasele in t:
        #         print(t)
            if t in seen_words:
                if t in non_convertables:
                    non_convertables[t] += 1
            else:
                candidates = transformer.transform(t, None)
                # if not candidates and any(t.startswith(pre) for pre in ['از', 'در', 'چند', 'هر', 'هیچ', 'هم', 'با', 'بی', 'تا', 'و']):
                #     print(t)
                if not candidates:
                    non_convertables[t] = 1
                seen_words.add(t)
    words_count = sorted([(word, count) for word, count in non_convertables.items()], key=lambda item: item[1], reverse=True)
    words_count = [str(word) + ' ########### ' + str(count) for (word, count) in words_count]
    with open(output_addr, 'w+') as f:
        f.write('\n'.join(words_count))


def generate_irrgular_informal_verbs():
    """
    برمیگرده میوفته برمیداره برمیگردونه درمیاره ایستادن نمیومد وامیسته

    اومد
    نیومد
    اومدی
    نیومدی
    میومدی
    نیومده
    یومد
    میومده
    """

    mapping_verbs = []
    past_ends = ['م', 'ی', 'ه', 'یم', 'ین', 'ید', 'ند', '', 'ن']
    neg = ['ن', '']
    pre = ['می', 'ب']
    pre_verbs = [('بر', 'دار'), ('در', 'یار'), ('وا', 'ست'), ('بر', 'گرد'), ('ور', 'دار'), ('بر', 'گشت')]
    extras = ['ن', 'نمی', 'می']
    mapper = {'ه':'د', 'ن': 'ند', 'ین': 'ید', 'ور': 'بر', 'ست':'ایست', 'وا':'', 'یار':'آور'}
    for item in pre_verbs:
        for pe in past_ends:
            for ex in extras:
                p_end = pe
                item0 = item[0]
                item1 = item[1]
                inf = item0 + ex + item1 + p_end
                inf = inf.replace('یی', 'ی')
                if item0 in mapper:
                    item0 = mapper[item0]
                if item1 in mapper:
                    item1 = mapper[item1]
                if p_end in mapper:
                    p_end = mapper[p_end]
                formal = item0 + ex + item1 + p_end
                formal = formal.replace('می', 'می‌')
                formal = formal.replace('نآ', 'نیا')
                mapping_verbs.append([formal, inf])
    bons = ['یومد', 'یوفت']
    v_mapper = {'یومد': 'یامد', 'یوفت': 'افت'}
    verbs = itertools.product(neg, pre, bons, past_ends)
    for v in verbs:
        if v[0] == 'ن' and v[1] == 'ب' or (v[2] == 'یومد' and v[1] == 'ب'):
            continue
        inf = v[0] + v[1] + v[2] + v[3]
        inf = inf.replace('یی', 'ی')
        pe = v[3]
        if pe in mapper:
            pe = mapper[pe]
        formal = v[0] + v[1]  +  '‌' + v_mapper[v[2]] + pe
        formal = formal.replace('ی‌ی', 'ی')
        formal = formal.replace('یا', 'ی‌آ')
        formal = formal.replace('دد', 'ده')
        formal = formal.replace('ب‌ا', 'بی')
        mapping_verbs.append([formal, inf])
    add_to_mapper(mapping_verbs)



def load_vocab(vocab_addr='resources/words.dat'):
    vocab = {}
    with open(vocab_addr, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                word, freq, p_tags = line.strip().split('\t')
                vocab[word] = {'freq': freq, 'tags': p_tags}
            except:
                word = line.strip()
                vocab[word] = {'freq': 1, 'tags': 'NUM'}
    return vocab

def if_connect(word1, word2):
    not_connect_chars = ['ا', 'د', 'ذ', 'ر', 'ز', 'ژ', 'و']
    if any(w =='' for w in [word1, word2]) or word1[-1] in not_connect_chars:
        return True
    return False
def split_conj_words(word, conjs):
    candidates = set()
    sorted_conjs = sorted(conjs, key=lambda x: len(x), reverse=True)
    for c in sorted_conjs:
        indx = word.find(c)
        if indx != -1 and indx in [0, len(word)-1]:
            pre_w = word[:indx]
            next_w = word[indx+len(c) :]
            if if_connect(pre_w, c) and if_connect(c, next_w):
                cnd = ' '.join([pre_w, c, next_w])
                cnd = cnd.strip()
                candidates.add(cnd)
    return list(candidates)


def is_formal_prefixed(word, vocab):
    not_connect_chars = ['ا', 'د', 'ذ', 'ر', 'ز', 'ژ', 'و']
    nim_fasele = '‌'
    m1 = re.match('(.+)های(م|ت|ش|مان|تان|شان)?$', word)
    m2 = re.match('(.+[ا|و|ی])ی(م|ت|ش|مان|تان|شان)$', word)
    m3 = re.match('(.+[^ا^و^ی])(م|ت|ش|مان|تان|شان)$', word)
    m4 = re.match('(.+)(ها)$', word)
    m5 = re.match('(.+[ه|ی]‌)(اش|ام|ات)$', word)
    if m3 or m2:
        prefix_word = list(filter(lambda m: m is not None, [m3, m2]))[0].group(1)
        if prefix_word in vocab:
            return True
    m_fired = list(filter(lambda m: m is not None, [m1, m4, m5]))
    if len(m_fired) > 0:
        # print(word, m_fired[0].groups())
        prefix_word = m_fired[0].group(1)
        if prefix_word[-1] != nim_fasele and prefix_word[-1] not in not_connect_chars:
            return False
        if prefix_word[-1] == nim_fasele and not (prefix_word[:-1] in vocab):
            return False
        if prefix_word[-1] != nim_fasele and not (prefix_word in vocab):
            return False
        return True
    return False


def spelling_similairty(word):
    all_possible = []
    possible_repeated = get_possible_repeated_word(word)
    all_possible = possible_repeated
    if word in all_possible:
        all_possible.remove(word)
    return all_possible

def add_nim_alef_hat_dictionary(vocab):
    word_with_hat = filter(lambda w: 'آ' in w, vocab)
    word_with_nim = filter(lambda w: '‌' in w, vocab)
    mapper1 = {w.replace('آ', 'ا').replace('‌', ''): w for w in word_with_hat}
    mapper2 = {w.replace('‌', ''): w for w in word_with_nim}
    mapper1.update(mapper2)
    return mapper1

def generate_spell_mapper(vocab):
    hat = 'آ'
    tanvin =  'اً'
    nim =  '‌'
    hamzeh = 'أ'
    hamzeh_y = 'ئ'
    sp_mapper = {hamzeh_y: ['ی'], hat: ['ا'], tanvin: ['ن', 'ا'], nim:['', ' '], hamzeh:['ا', '']}
    special_chars = [hat, tanvin, nim, hamzeh]
    out = {}
    for word in vocab:
        p_words = [word.replace(sp, sp_alt) for sp in special_chars for sp_alt in sp_mapper[sp]]
        spell_errors = []
        p_words = list(set(p_words) - set([word]))
        for pw in p_words:
            if pw in out:
                out[pw].add(word)
            else:
                out[pw] = {word}
    out = {w: list(out[w]) for w in out}
    with open('spell_checker_mapper.json', 'w+', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)



def create_mapper_tanvin_hamze_hat_nim_fasele():
    mapper = {}
    hats_word = open('resources/spell/words_with_hat.txt').read().splitlines()
    nim_words = open('resources/spell/words_with_nim.txt').read().splitlines()
    tanvin_words = open('resources/spell/words_with_tanvin.txt').read().splitlines()
    hat_ch = 'آ'
    nim_fasele = '‌'
    for w in hats_word:
        w_without_h = w.replace(hat_ch, 'ا')
        mapper[w_without_h] = w
    for w in nim_words:
        w_without_nim = w.remove(nim_fasele)
        mapper[w_without_nim] = w
        w_space_instead_nim = w.replace(nim_fasele, ' ')
        mapper[w_space_instead_nim] = w

def extract_lemma_nim_fasele_words(word, vocab):
        prefixs = ['اون']
        postfixs = {'ست': 'است', 'هام':'هایم', 'ام':'ام', 'ها':'ها', 'هامون':'هایمان', 'ترین': 'ترین', 'هایشان':'هایشان'}
        tokens = word.split('‌')
        index = 0
        for i in range(len(tokens)):
            index = i
            if tokens[i] not in prefixs:
                break

        for i in range(len(tokens), 0, -1):
            current_tok = '‌'.join(tokens[index:i])
            if current_tok in vocab or  tokens[i-1] not in postfixs:
                return current_tok


def if_emoji(text):
    # Wide UCS-4 build
    try:
        oRes = re.compile(u'(['
                          u'\U0001F300-\U0001F64F'
                          u'\U0001F680-\U0001F6FF'
                          u'\u2600-\u26FF\u2700-\u27BF]+)',
                          re.UNICODE)

    except re.error:
        # Narrow UCS-2 build
        oRes = re.compile(u'(('
                          u'\ud83c[\udf00-\udfff]|'
                          u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
                          u'[\u2600-\u26FF\u2700-\u27BF])+)',
                          re.UNICODE)

    return oRes.findall(text)


def powerset(lst):
    return reduce(lambda result, x: result + [subset + [x] for subset in result],
                  lst, [[]])