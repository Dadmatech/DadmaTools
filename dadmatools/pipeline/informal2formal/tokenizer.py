import itertools
import dadmatools.pipeline.informal2formal.utils as utilss


class InformalTokenizer:
    def __init__(self, vocab, postfixes):
        self.vocab = vocab
        self.pres = InformalTokenizer.get_prefixs()
        self.posts = postfixes

    @staticmethod
    def get_prefixs():
        return ['نا', 'بی', 'هر', 'می']

    @staticmethod
    def get_postfixs(informal_postfix_addr):
        with open(informal_postfix_addr, 'r') as f:
            ps = f.read().splitlines()
        return ps

    def is_pre_post_word(self, w):
        nim_fasele = '‌'
        ws = w.split(nim_fasele)
        pre, pos, v = [0, 1, 2]
        is_pre_pos = False
        state = pre
        valid_w = ''
        for w in ws:
            if state == pre:
                if w in self.pres:
                    valid_w += nim_fasele + w
                    is_pre_pos = True
                    continue
                elif w in self.posts:
                    valid_w += nim_fasele + w
                    is_pre_pos = True
                    state = pos
                    continue
                state = v
                valid_w += nim_fasele + w
                continue

            if state == pos:
                if w in self.posts:
                    valid_w += nim_fasele + w
                    continue
                return False
            if state == v:
                if w in self.posts:
                    is_pre_pos = True
                    state = pos
                    valid_w += nim_fasele + w
                    continue
                if w in self.vocab:
                    valid_w += nim_fasele + w
                    if valid_w not in self.vocab:
                        return False
                    continue

                return False
        if not is_pre_pos:
            return False
        return True

    def get_valid_word(self, words):
        seps = ['', '‌']
        all_seqs = []
        count = len(words)
        lst = list(itertools.product(seps, repeat=count - 1))
        for item in lst:
            seq = ''
            for word, sep in zip(words[:-1], item):
                seq += word + sep
            seq += words[-1]
            all_seqs.append(seq)
        return [w for w in all_seqs if w in self.vocab or self.is_pre_post_word(w)]

    def get_candidates(self, tokens, index=0, current_seq=' '):
        if index == len(tokens):
            return current_seq
        word = tokens[index]
        next_word, next_next_word = [None, None]
        if index < len(tokens) - 1:
            next_word = tokens[index + 1]
        if index < len(tokens) - 2:
            next_next_word = tokens[index + 2]
        cnds = []
        if next_word is not None:
            v_words = self.get_valid_word([word, next_word])
            if v_words:
                for v_w in v_words:
                    current_seq1 = current_seq + ' ' + v_w
                    cnds2 = self.get_candidates(tokens, index + 2, current_seq1)
                    if type(cnds2) == str:
                        cnds.append(cnds2)
                    else:
                        cnds.extend(cnds2)
        if next_next_word is not None:
            v_words = self.get_valid_word([word, next_word, next_next_word])
            if v_words:
                for v_w in v_words:
                    current_seq2 = current_seq + ' ' + v_w
                    cnds3 = self.get_candidates(tokens, index + 3, current_seq2)
                    if type(cnds3) == str:
                        cnds.append(cnds3)
                    else:
                        cnds.extend(cnds3)
        current_seq = current_seq + ' ' + word
        cnds1 = self.get_candidates(tokens, index + 1, current_seq)
        if type(cnds1) == str:
            cnds.append(cnds1)
        else:
            cnds.extend(cnds1)
        return [c.strip() for c in cnds]

    def if_connect(word1, word2):
        not_connect_chars = ['ا', 'د', 'ذ', 'ر', 'ز', 'ژ', 'و']
        if any(w == '' for w in [word1, word2]) or word1[-1] in not_connect_chars:
            return True
        return False

    def seperate_conjs(self, word, validator):
        conjs = ['و', 'در', 'با', 'تا', 'که', 'از', 'تو', 'من', 'شما']
        cnds = utilss.split_conj_words(word, conjs)
        valid_cnds = [c for c in cnds if validator(c)]
        if valid_cnds:
            return valid_cnds
        return [word]

    def tokenize(self, txt, validator):
        tokens = txt.split()
        all_cnds = []
        for t in tokens:
            if not validator(t):
                ws = self.seperate_conjs(t, validator)
            else:
                ws = [t]
            all_cnds.append(ws)
        all_cnd_tokens = itertools.product(*all_cnds)
        txts = list(map(self.get_dense_tokens, all_cnd_tokens))
        return txts

    def get_dense_tokens(self, tokens):
        PRE, WORD, POST = 0, 1, 2
        out_tokens = []
        nim_fasele = '‌'
        current_word = ''
        state = WORD
        for i, t in enumerate(tokens):
            if state == WORD:
                if t in self.pres:
                    out_tokens.append(current_word)
                    current_word = t
                    state = PRE
                if t in self.posts:
                    current_word += nim_fasele
                    current_word += t
                    state = POST
                if t not in self.pres and t not in self.posts:
                    out_tokens.append(current_word)
                    current_word = t
                continue
            if state == PRE:
                if t in self.pres:
                    current_word += nim_fasele
                    current_word += t
                if t in self.posts:
                    out_tokens.append(current_word)
                    current_word = t
                    state = WORD
                if t not in self.pres and t not in self.posts:
                    current_word += nim_fasele
                    current_word += t
                    state = WORD
                continue
            if state == POST:
                if t in self.pres:
                    out_tokens.append(current_word)
                    current_word = t
                    state = PRE
                if t in self.posts:
                    current_word += nim_fasele
                    current_word += t
                if t not in self.pres and t not in self.posts:
                    out_tokens.append(current_word)
                    current_word = t
                    state = WORD
        if out_tokens[-1] != current_word:
            out_tokens.append(current_word)
        return out_tokens
