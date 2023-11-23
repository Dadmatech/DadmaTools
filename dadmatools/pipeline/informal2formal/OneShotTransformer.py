import re
import itertools
import string
import dadmatools.pipeline.informal2formal.utils as utils


class InformalWord:
    def __init__(self, lemma, prefixs=None, postfixs=None, pos=None, append_h=False):
        if prefixs is None:
            prefixs = []
        if postfixs is None:
            postfixs = []
        self.is_verb = False
        self.is_mapper = False
        self.semi_mapper = False
        self.append_h = append_h
        self.lemma = lemma
        self.prefixs = prefixs
        self.postfixs = postfixs
        self.pos = pos

class Prefix:
    def __init__(self, word, level, formal=None, ignore_poses=None, poses=None, non_connecting_chars=None, connector='nim'):
        if non_connecting_chars is None:
            non_connecting_chars = []
        self.word = word
        self.level = level
        self.ignore_poses = ignore_poses
        self.poses = poses
        self.connector = connector
        if formal is None:
            self.formal = word
        else:
            self.formal = formal
        self.non_connecting_chars = non_connecting_chars
class Postfix:
    def __init__(self, word, level, formal=None, ignore_poses=None, non_connecting_chars=None, poses=None, connector='nim'):
        if non_connecting_chars is None:
            non_connecting_chars = []
        self.word = word
        self.level = level
        self.ignore_poses = ignore_poses
        self.poses = poses
        self.connector = connector
        if formal is None:
            self.formal = word
        else:
            self.formal = formal
        self.non_connecting_chars = non_connecting_chars



class OneShotTransformer:

    NIM_FASELE = chr(8204)
    # prefixs
    HAMUN = Prefix('همون', 1, 'همان',connector='fasele',non_connecting_chars=['ه'])
    HAMIN = Prefix('همین', 1,connector='fasele')
    HAR = Prefix('هر', 1,connector='fasele')
    UN = Prefix('اون', 1, 'آن',connector='fasele',non_connecting_chars=['ه'])
    IN = Prefix('این', 1,connector='fasele',non_connecting_chars=['ه'])
    HICH = Prefix('هیچ', 1,connector='nim',non_connecting_chars=['ه', 'ا', 'آ'])
    B = Prefix('ب', 1, 'به', ignore_poses=['VERB', 'CCONJ', 'SCONJ'],connector='fasele',non_connecting_chars=['ا', 'ه', 'آ'])
    Y = Prefix('ی', 1, 'یک', ignore_poses=['VERB', 'CCONJ', 'SCONJ'],connector='fasele',non_connecting_chars=['ا', 'آ'])
    BI = Prefix('بی', 1, ignore_poses=['VERB'],connector='nim',non_connecting_chars=['ا'])
    POR = Prefix('پر', 1, ignore_poses=['VERB'],connector='nim')
    pres = [[HAMIN, HAMUN, UN, IN, HAMIN, BI, B, Y, POR, HAR]]
    #postfixs
    Y1 = Postfix('ی', 0, ignore_poses=['VERB'], connector='none',non_connecting_chars=['ی', 'ا', 'و', 'آ', 'اً'])
    TAR = Postfix('تر', 1, connector='nim')
    TARIN = Postfix('ترین', 1, connector='nim')
    HAY = Postfix('های', 2, connector='nim')
    HA = Postfix('ها', 2, connector='nim')
    A = Postfix('ا', 2, 'ها', ignore_poses=['VERB'], connector='nim',non_connecting_chars=['ا', 'و', 'آ', 'اً'])
    A1 = Postfix('ای', 2, 'های', ignore_poses=['VERB'], connector='nim',non_connecting_chars=['ا', 'و', 'آ', 'اً'])
    YY = Postfix('یی', 3, 'یی', ignore_poses=['VERB'], connector='none')
    M = Postfix('م', 3, ignore_poses=['VERB'], connector='none')
    M_MAN = Postfix('م', 3, 'من', ignore_poses=['VERB'], connector='fasele')
    T = Postfix('ت', 3, connector='none')
    T1 = Postfix('ت', 3, 'تو', connector='fasele')
    # T2 = Postfix('ت', 3, 'خود', ignore_poses=['VERB'], connector='fasele')
    SH = Postfix('ش', 3, connector='none')
    # SH1 = Postfix('ش', 3, 'خود', connector='fasele')
    # SH2 = Postfix('ش', 3, 'آن', connector='fasele')
    # SH3 = Postfix('ش', 3, 'او', connector='fasele')
    MAN = Postfix('مان', 3, connector='nim')
    MAN1 = Postfix('مان', 3, 'ما', connector='fasele')
    # MAN2 = Postfix('مان', 3, 'خود', connector='fasele')
    MUN = Postfix('مون', 3, 'مان', connector='nim')
    # MUN1 = Postfix('مون', 3, 'خود', connector='fasele')
    MUN2 = Postfix('مون', 3, 'ما', connector='fasele')
    TAN = Postfix('تان', 3, connector='nim')
    # TAN1 = Postfix('تان', 3, 'خود', connector='fasele')
    TAN2 = Postfix('تان', 3, 'شما', connector='fasele')
    TUN = Postfix('تون', 3, 'تان', connector='nim')
    # TUN1 = Postfix('تون', 3, 'خود', connector='fasele')
    TUN2 = Postfix('تون', 3, 'شما', connector='fasele')
    SHAN = Postfix('شان', 3, connector='nim')
    # SHAN1 = Postfix('شان', 3, 'خود', connector='fasele')
    SHAN2 = Postfix('شان', 3, 'آنان', connector='fasele')
    SHUN = Postfix('شون', 3, 'شان', connector='nim')
    # SHUN1 = Postfix('شون', 3, 'خود', connector='fasele')
    SHUN2 = Postfix('شون', 3, 'آنان', connector='fasele')
    N = Postfix('ن', 4, 'هستند', ignore_poses=['VERB', 'CCONJ', 'SCONJ'], connector='fasele', non_connecting_chars=['ی'])
    SHAM = Postfix('شم', 4, 'بشوم',ignore_poses=['VERB'], connector='fasele')
    SHI= Postfix('شی', 4, 'بشوی',ignore_poses=['VERB'], connector='fasele')
    SHE= Postfix('شه', 4, 'شود',ignore_poses=['VERB'], connector='fasele')
    SHIN= Postfix('شین', 4, 'شوید',ignore_poses=['VERB'], connector='fasele')
    SHID= Postfix('شید', 4, 'شوید',ignore_poses=['VERB'], connector='fasele')
    SHAAN= Postfix('شن', 4, 'شوند',ignore_poses=['VERB'], connector='fasele')
    SHAND= Postfix('شند', 4, 'شوند',ignore_poses=['VERB'], connector='fasele')
    M2 = Postfix('م', 4, 'هم',ignore_poses=['VERB'], connector='fasele')
    V = Postfix('و', 4, 'را', connector='fasele', non_connecting_chars=['ا', 'ای', 'آ', 'اً'])
    V1 = Postfix('رو', 4, 'را', connector='fasele')
    H = Postfix('ه', 4, '', ignore_poses=['VERB', 'CCONJ', 'SCONJ'], connector='none')
    # H2 = Postfix('ه', 4)
    M1 = Postfix('م', 4, 'هستم',ignore_poses=['VERB'], connector='fasele')
    Y2 = Postfix('ی', 4, 'ی', ignore_poses=['VERB'], connector='none')
    H1 = Postfix('ه', 4, 'است', ignore_poses=['VERB'], connector='fasele', non_connecting_chars=['ا', 'آ', 'اً'])
    S = Postfix('س', 4, 'است', connector='fasele')
    ST = Postfix('ست', 4, 'است', connector='fasele')
    ED = Postfix('ید', 4, 'هستید', ignore_poses=['VERB'], connector='fasele')
    EN = Postfix('ین', 4, 'هستید', ignore_poses=['VERB'], connector='fasele', non_connecting_chars=['تر'])
    EM = Postfix('یم', 4, 'هستیم', ignore_poses=['VERB'], connector='fasele')
    ND = Postfix('ند', 4, 'هستند', ignore_poses=['VERB'], connector='fasele')
    # posts = [[Y1], [TAR, TARIN], [HA, HAY, A, A1], [M, T, SH, MAN, MUN, TAN, TUN, SHAN, SHUN], [N, S, ST, M1, M2, V, V1,Y2, H, H1, ED, EN, EM, ND, SHAM, SHI, SHID, SHE, SHAND, SHIN, SHAAN]]
    # posts = [[Y1], [TAR, TARIN], [HA, HAY, A, A1], [YY, M, M_MAN, T, T1, T2, SH, MAN, MAN1, MAN2,MUN,MUN1,MUN2, TAN,TAN1,TAN2, TUN,TUN1,TUN2, SHAN,SHAN1,SHAN2, SHUN, SHUN1, SHUN2], [N, S, ST, M1, M2, V, V1,Y2, H1, ED, EN, EM, ND, SHAM, SHI, SHID, SHE, SHAND, SHIN, SHAAN]]
    posts = [[Y1], [TAR, TARIN], [HA, HAY, A, A1], [YY, M, M_MAN, T, T1,  SH, MAN, MAN1,MUN,MUN2, TAN,TAN2, TUN,TUN2, SHAN,SHAN2, SHUN, SHUN2], [N, S, ST, M1, M2, V, V1,Y2, H1, ED, EN, EM, ND, SHAM, SHI, SHID, SHE, SHAND, SHIN, SHAAN]]
    PossessiveـPronouns = [M,T,SH, MAN, MUN, TAN, TUN, SHAN, SHUN]
    cant_append_h_posts = [Y1, TAR, TARIN]
    As = [A, A1]

    def get_separator(self, w1, w2, append_h):
        connector_2_str = {'none': '', 'nim': OneShotTransformer.NIM_FASELE, 'fasele': ' '}
        not_connect_chars = ['ا', 'د', 'ذ', 'ر', 'ز', 'ژ', 'و']
        # if w2 == OneShotTransformer.Y2:
        #     return ''
        # if w2 in [OneShotTransformer.M, OneShotTransformer.T, OneShotTransformer.SH] and ( type(w1) == str and w1[-1] in ['ا', 'و']):
        #     return 'ی'
        # if type(w1) != str and w1.level == 1:
        #     return ' '
        # not_connect_chars = ['ا', 'د', 'ذ', 'ر', 'ز', 'ژ', 'و']
        # if w1 in [OneShotTransformer.Y, OneShotTransformer.B, OneShotTransformer.HAMIN, OneShotTransformer.IN, OneShotTransformer.HAMUN] or w2 in [OneShotTransformer.ED, OneShotTransformer.EN, OneShotTransformer.EM, OneShotTransformer.ND, OneShotTransformer.H1, OneShotTransformer.M1, OneShotTransformer.S, OneShotTransformer.ST, OneShotTransformer.V, OneShotTransformer.N, OneShotTransformer.M2]:
        #     return ' '
        #
        # if ((type(w1) == str and len(w1)> 0 and w1[-1] in ['ا', 'و']) or (type(w1) != str and  w1.formal[-1] in [ 'ا', 'و']))and w2.level == 3 :
        #     return 'ی' + '‌'
        # if (type(w1) == str and len(w1)> 0 and w1[-1] in not_connect_chars) or (type(w1) != str and w1.word[-1] in not_connect_chars):
        #     return ''
        all_pres = [p for pres in OneShotTransformer.pres for p in pres]
        all_posts = [p for posts in OneShotTransformer.posts for p in posts]
        if type(w1) == str:
            last_ch = w1[-1]
        else:
            last_ch = w1.word[-1]
        separator = ''
        extra_sep = ''
        if type(w1) == str and append_h and w2 in [OneShotTransformer.M, OneShotTransformer.T, OneShotTransformer.SH]:
            extra_sep = OneShotTransformer.NIM_FASELE + 'ا'
        if w2 in [OneShotTransformer.M, OneShotTransformer.T, OneShotTransformer.SH, OneShotTransformer.MAN, OneShotTransformer.MUN, OneShotTransformer.TAN, OneShotTransformer.TUN, OneShotTransformer.SHAN, OneShotTransformer.SHUN] and ( last_ch in ['ا', 'و']) :
            extra_sep = 'ی'
        if w1 in all_pres:
            separator = connector_2_str[w1.connector]
        if w2 in all_posts:
            separator = connector_2_str[w2.connector]

        # replace nim_fasele with '' for non connected words

        if last_ch in not_connect_chars and separator == OneShotTransformer.NIM_FASELE:
            separator = ''
        return extra_sep + separator

    def lemma_to_formals(self, iword):
        out_iwords = [iword]
        if iword.lemma in self.mapper and self.iword2str(iword) != self.mapper[iword.lemma]:
            for map_words in self.mapper[iword.lemma]:
                new_iw = InformalWord(lemma=map_words,prefixs=iword.prefixs, postfixs=iword.postfixs, pos=iword.pos, append_h=iword.append_h)
                if not iword.prefixs and not iword.postfixs:
                    new_iw.is_mapper = True
                    new_iw.semi_mapper = True
                else:
                    new_iw.semi_mapper = True
                out_iwords.append(new_iw)
        formal_verbs = self.verb_to_formal_func(iword.lemma)
        if formal_verbs is not None:
            for f_v in formal_verbs:
                new_iw = InformalWord(lemma=f_v,prefixs=iword.prefixs, postfixs=iword.postfixs, pos=iword.pos, append_h=iword.append_h)
                new_iw.is_verb = True
                out_iwords.append(new_iw)
        return out_iwords


    def should_ignore_by_postagg(self, iword):
        post_pres = [pre for pre in iword.prefixs] + [post for post in iword.postfixs]
        for p in post_pres:
            if (p.ignore_poses and iword.pos in p.ignore_poses) or (p.poses and iword.pos not in p.poses):
                return True
        return False

    def filtered_based_on_rules(self, iword):
        #YY
        ha_p = [OneShotTransformer.A, OneShotTransformer.HA]
        if iword.postfixs and OneShotTransformer.YY in iword.postfixs and not all(p in ha_p + [OneShotTransformer.YY] for p in iword.postfixs):
            return True
        #hasti!
        if (iword.postfixs and len(iword.postfixs) == 1 and OneShotTransformer.Y2 in iword.postfixs and iword.lemma and iword.lemma[-1] in ['و', 'ا']) or (iword.postfixs and len(iword.postfixs) == 2 and OneShotTransformer.Y2 in iword.postfixs and iword.postfixs[0] in [OneShotTransformer.A, OneShotTransformer.HA]):
            return True
        #non connecting chars
        if iword.prefixs:
            last_pre = iword.prefixs[-1]
            if last_pre.non_connecting_chars and iword.lemma and any(iword.lemma.startswith(ch) for ch in last_pre.non_connecting_chars):
                return True
        if iword.postfixs:
            first_post = iword.postfixs[0]
            if first_post.non_connecting_chars and iword.lemma and any(iword.lemma.endswith(ch) for ch in first_post.non_connecting_chars):
                return True
        #hidden H # goshnashe
        if not iword.semi_mapper and not iword.append_h and iword.lemma and iword.lemma[-1] == 'ه' and iword.postfixs  and iword.lemma not in self.non_hidden_h_words:
            return True
        # h + h
        if iword.prefixs and iword.postfixs and len(iword.lemma) < 2:
            return True
        # خونهه - خونششونه
        if iword.append_h and (OneShotTransformer.H in iword.postfixs or (len(iword.postfixs) == 1 and OneShotTransformer.H1 in iword.postfixs) ):
           return True
        if iword.prefixs and (OneShotTransformer.B in iword.prefixs or OneShotTransformer.Y in iword.prefixs) and (iword.lemma and iword.lemma[0] in ['ا', 'ی', 'و']):
            return True
        if iword.lemma in self.isolated_words and (iword.prefixs or iword.postfixs):
            return True
        # verb + postfixs ex:  برنامه
        if (iword.is_verb and iword.prefixs) or(iword.is_verb and iword.postfixs and (len(iword.postfixs) > 1 or not any(p in iword.postfixs for p in OneShotTransformer.PossessiveـPronouns +[OneShotTransformer.V]))):
            return True
        return False

    def iword2str(self, iword):
        sorted_prefixs = list(sorted(iword.prefixs, key=lambda prefix: prefix.level))
        sorted_postfixs = list(sorted(iword.postfixs, key=lambda postfix: postfix.level))
        concated_str = ''
        zipped_prefixs = [(sorted_prefixs[i], sorted_prefixs[i + 1]) if i < len(sorted_prefixs) - 1 else (
        sorted_prefixs[i], iword.lemma) for i in range(len(sorted_prefixs))]
        for prev_prefix, prefix in zipped_prefixs:
            separator = self.get_separator(prev_prefix, prefix, append_h=False)
            prefix_formal = prev_prefix.formal
            concated_str += prefix_formal
            concated_str += separator

        concated_str += iword.lemma

        zipped_postfix = [(sorted_postfixs[i - 1], sorted_postfixs[i]) if i > 0 else (iword.lemma, sorted_postfixs[i])
                          for i in range(len(sorted_postfixs))]
        for postfix, next_postfix in zipped_postfix:
            separator = self.get_separator(postfix, next_postfix, append_h=iword.append_h)
            concated_str += separator
            postfix_formal = next_postfix.formal
            concated_str += postfix_formal
        return concated_str

    def to_formals(self, iword):
        str_iwords = []
        all_iwords = self.lemma_to_formals(iword)
        for iword in all_iwords:
            # if iword.lemma == 'اون':
            #     print('')
            if len(iword.lemma) == 1 and iword.lemma != 'و':
                str_iwords.append(('', None))
                continue
            if self.filtered_based_on_rules(iword):
                str_iwords.append(('', None))
                continue
            if self.should_ignore_by_postagg(iword):
                str_iwords.append(('', None))
                continue
            if not iword.is_verb and not iword.semi_mapper and iword.lemma not in self.vocab:
                str_iwords.append(('', None))
                continue
            concated_str = self.iword2str(iword)
            str_iwords.append((concated_str, iword))
        return str_iwords

    def un_in(self, iword):
        new_lemma = iword.lemma.replace('ون', 'ان')
        if new_lemma != iword.lemma:
            return InformalWord(lemma=new_lemma, prefixs=iword.prefixs, postfixs=iword.postfixs, pos=iword.pos)
        else:
            return False

    def prefix_obj(self, word):
        op_separete = {'م': 'من', 'ت': 'تو', 'ش': 'آن', 'تان': 'شما', 'تون': 'شما', 'شون': 'آنان', 'شان': 'آنان',
                       'مان': 'ما', 'مون': 'ما'}
        candidates = []
        formal = ''
        m = self.pre_obj_pattern.match(word)
        if m:
            tokens = m.groups()
            if tokens[0] == 'باها':
                formal += 'با'
            else:
                formal += tokens[0]
            formal_obj = op_separete[tokens[1]]
            formal += ' '
            formal += formal_obj
            if tokens[2] is not None:
                formal += ' '
                formal += 'هم'
            alts = {'هم': 'هستم', 'آن': 'او'}
            tokens = [[w] for w in formal.split()]
            for t in tokens:
                if t[0] in alts:
                    t.append(alts[t[0]])

            candidates = itertools.product(*tokens)
            candidates = [' '.join(cnd) for cnd in candidates]

        return [(c, c) for c in candidates]



    def append_tanvin_hat(self, iword):
        if len(iword.lemma) > 1 and iword.lemma[0] == 'ا' and iword.lemma[-1] != 'ا':
            new_lemma = 'آ' + iword.lemma[1:]
            return InformalWord(lemma=new_lemma, prefixs=iword.prefixs, postfixs=iword.postfixs, pos=iword.pos)
        if len(iword.lemma) > 1 and iword.lemma[-1] == 'ا':
            new_lemma = iword.lemma[:-1] + 'اً'
            return InformalWord(lemma=new_lemma, prefixs=iword.prefixs, postfixs=iword.postfixs, pos=iword.pos)
        return False

    def append_h(self, iword):
        not_apply = self.verb_to_formal_func(iword.lemma) or (iword.lemma and iword.lemma[-1] in ['ا', 'و', 'ی'])  or len(iword.lemma) <= 1 or iword.lemma =='' or iword.lemma[-1] == 'ه' or (OneShotTransformer.H in iword.postfixs and len(iword.postfixs) == 1) or any(p in iword.postfixs for p in OneShotTransformer.As) or(OneShotTransformer.V in iword.postfixs) or (iword.postfixs and iword.postfixs[0].word[0] in ['ی', 'و','ا'])
        ######## when add h?
        new_lemma = iword.lemma + 'ه'
        ############# new_lemma in self.vocab
        if len(iword.postfixs) > 0 and not any([p in OneShotTransformer.cant_append_h_posts for p in iword.postfixs]) and not not_apply and new_lemma not in self.non_hidden_h_words:
        # if len(iword.postfixs) > 0 and not not_apply and new_lemma in self.vocab and new_lemma not in self.non_hidden_h_words:
            return InformalWord(lemma=new_lemma, prefixs=iword.prefixs, postfixs=iword.postfixs, pos=iword.pos, append_h= True)
        return False

    def __init__(self, vocab, mapper, verb_to_formal_func, ignore_words, postfix_mapper, isolated_words, non_hidden_h_words):
        self.vocab = vocab
        self.mapper = mapper
        self.verb_to_formal_func = verb_to_formal_func
        self.ignore_words = ignore_words
        self.postfix_mapper = postfix_mapper
        self.isolated_words = isolated_words
        self.non_hidden_h_words = non_hidden_h_words
        self.operators = [self.un_in, self.append_h, self.append_tanvin_hat]
        patt = r'(از|به|باها)(مان|شون|شان|مون|م|تون|تان|ت|ش)(م)?$'
        self.pre_obj_pattern = re.compile(patt)

    def all_sequence_of_postfixs(self, word, index):
        all_seqs  =[]
        for p in OneShotTransformer.posts[index]:
            p_w = p.word
            if word.startswith(p_w):
                w = word[len(p_w):]
                if len(w) == 0:
                    all_seqs.append(p)
                else:
                    if index < len(OneShotTransformer.posts) -1 :
                        resp = self.all_sequence_of_postfixs(w, index+1)
                        if len(resp) > 0:
                            for item in resp:
                                if type(item) == list:
                                    item.append(p)
                                    sequence_with_p = item
                                else:
                                    sequence_with_p = [p, item]
                                all_seqs.append(sequence_with_p)
        if index < len(OneShotTransformer.posts) - 1:
            resp = self.all_sequence_of_postfixs(word, index + 1)
            all_seqs.extend(resp)
        else:
            return all_seqs
        return all_seqs

    def combine(self, l1, l2):
        if len(l1) == 0:
            return l2
        elif len(l2) == 0:
            return l1
        return list(itertools.product(l1, l2))


    def get_expand(self, iword):
        all_possible_words = []
        for subset_operators in utils.powerset(self.operators):
            new_iword = InformalWord(lemma=iword.lemma, prefixs=iword.prefixs, postfixs=iword.postfixs, pos=iword.pos)
            for so in subset_operators:
                so_resp = so(new_iword)
                if so_resp:
                    new_iword = so_resp
            all_possible_words.append(new_iword)
        return all_possible_words


    def match_postfixs(self, word, pos):
        possible_combinatios = []
        for i in range(len(OneShotTransformer.posts)):
            for p in OneShotTransformer.posts[i]:
                p_word = p.word
                p_indxs = [indx for indx, ch in enumerate(word) if word[indx:indx+len(p_word)] == p_word]
                for p_indx in p_indxs:
                    if p_indx != -1:
                        lemma = word[:p_indx]
                        pp = word[p_indx + len(p_word):]
                        if len(pp) ==0:
                            iw = InformalWord(lemma=lemma, postfixs=[p], pos=pos)
                            possible_combinatios.append(iw)
                            continue
                        if i < len(OneShotTransformer.posts) -1:
                            all_postfix = self.all_sequence_of_postfixs(pp, index=i+1)
                            if len(all_postfix) > 0:
                                for pfixs in all_postfix:
                                    if type(pfixs) == list:
                                        pfixs.append(p)
                                    else:
                                        pfixs = [p, pfixs]
                                    iw = InformalWord(lemma=lemma, postfixs=pfixs, pos=pos)
                                    possible_combinatios.append(iw)
                        elif len(pp) == 0:
                            iw = InformalWord(lemma=lemma, postfixs=[p], pos=pos)
                            possible_combinatios.append(iw)

        return possible_combinatios

    def match_prefixs(self, word, pos):
        possible_combinatios = []
        for i in range(len(OneShotTransformer.pres)):
            for p in OneShotTransformer.pres[i]:
                if word.startswith(p.word):
                    lemma = word[len(p.word):]
                    prefixs = [p]
                    iw = InformalWord(lemma=lemma, prefixs=prefixs, postfixs=[], pos=pos)
                    possible_combinatios.append(iw)
                    return possible_combinatios
        return []

    def parse_word(self, iword):
        parsed_resp = []
        prefixed_word = self.match_prefixs(iword.lemma,pos=iword.pos)
        prefixed_word.append(iword)
        parsed_resp.extend(prefixed_word)
        for pw in prefixed_word:
            postfixed_iwords = self.match_postfixs(pw.lemma,pos=iword.pos)
            for piw in postfixed_iwords:
                piw.prefixs = pw.prefixs
                parsed_resp.append(piw)
        return parsed_resp

    def is_seqs_of_verbs(self, txt):
        words = txt.split()
        if len(words) < 2:
            return False
        for w in words:
            formal_verb = self.verb_to_formal_func(w)
            if formal_verb is None:
                return False
        if words[-1] in ['است', 'هست']:
            return False
        return True

    def filter_results(self, word_lemmas):
        return list(filter(lambda wl: len(wl[0])>0 and wl[0][-1] != '‌' and not self.is_seqs_of_verbs(wl[0]), word_lemmas))

    def concatenate_formal_words(self, pre, next):
        """
        خانه +‌ ت -> خانه‌ات
        دیگر + ای -> دیگری
        """
        nim_fasele = '‌'
        not_connect_chars = ['ا', 'د', 'ذ', 'ر', 'ز', 'ژ', 'و']
        if len(pre) < 1 :
            return next
        if pre[-1] in ['ه'] and next in ['م', 'ت', 'ش']:
            return pre + nim_fasele + 'ا' + next
        if pre[-1] == 'ا'and next.split() and next.split()[0] in ['م', 'ت', 'ش', 'مان', 'تان', 'شان']:
            return pre + nim_fasele + 'ی' + next
        if pre[-1] not in ['ه'] and next in ['ای']:
            return pre + 'ی'
        out = pre  + next
        if pre[-1] not in not_connect_chars or next.startswith('ها') or pre[-1] in ['ه'] or pre + nim_fasele + next in self.vocab:
            out = pre + nim_fasele + next
        if self.verb_to_formal_func(next):
            out = pre + ' ' + next
        return out

    def handle_nim_fasele_words(self, word, pos):
        def extract_lemma_nim_fasele_words(word, pos):
            formal_prefixs = []
            formal_postfixs = []
            prefixs = {'اون': 'آن', 'همون': 'همین'}
            postfixs = self.postfix_mapper
            tokens = word.split('‌')
            index = 0
            for i in range(len(tokens)):
                index = i
                if tokens[i] not in prefixs:
                    break
                else:
                    formal_prefixs.append(prefixs[tokens[i]])

            for i in range(len(tokens), index, -1):
                current_tok = '‌'.join(tokens[index:i])
                if current_tok in self.vocab or tokens[i - 1] not in postfixs:
                    return formal_prefixs, current_tok, formal_postfixs
                else:
                    formal_postfixs.append(postfixs[tokens[i - 1]])
            return formal_prefixs, current_tok, formal_postfixs
        nim_fasele = '‌'
        candidates = []
        formal_word = ''
        verbs = self.verb_to_formal_func(word)
        if verbs:
            return [(v, v) for v in verbs]
        all_candidates = set()
        # lemma
        formal_prefixs, lemma, formal_postfixs = extract_lemma_nim_fasele_words(word, pos)
        word_lemmas = self.transform(lemma, pos, ignore_nim_fasele=True)
        # lemma with postfix should len=1
        one_token_words = [wl for wl in word_lemmas if len(wl[0].split()) == 1]
        if formal_postfixs and one_token_words:
            all_formal_lemma_candidates = one_token_words
        else:
            all_formal_lemma_candidates = word_lemmas
        if not all_formal_lemma_candidates:
                if formal_postfixs or formal_prefixs:
                    all_formal_lemma_candidates = [(lemma, lemma)]
                else:
                    tokens = lemma.split(nim_fasele)
                    if all(self.transform(t, None, ignore_nim_fasele=True) for t in tokens):
                        w = ' '.join(tokens)
                        return [(w, w)]
                    else:
                        return []
        for cnd_lemma, formal_word_lemma in all_formal_lemma_candidates:
            formal_word = ''
            toks = formal_prefixs + [cnd_lemma] + formal_postfixs
            for index, t in enumerate(toks):
                formal_word = self.concatenate_formal_words(formal_word, t)
            all_candidates.add((formal_word, formal_word_lemma))
            #     if t in self.postfix_mapper:
            #         formal_t = self.postfix_mapper[t]
            #     else:
            #         transform_outputs = self.transform(t, pos)
            #         if not transform_outputs:
            #             formal_t = t
            #         else:
            #             one_word_outputs = [ft for ft in transform_outputs if len(ft.split()) == 1]
            #             if one_word_outputs:
            #                 if t in one_word_outputs:
            #                     formal_t = t
            #                 else:
            #                     formal_t = one_word_outputs[0]
            #             else:
            #                 formal_t = transform_outputs.pop()
        return all_candidates



    def transform(self, word, pos, ignore_nim_fasele=False):
        """ignore emoji , punctuation, numbers"""
        ignore_chars = '.1234567890!@#$%^&*()_+۱۲۳۴۵۶۷۸۹÷؟×−+?><}،,{":' + string.ascii_lowercase + string.ascii_uppercase
        if any(ic in word for ic in ignore_chars) or utils.if_emoji(word):
            return [(word, word)]
        """handle nim fasele"""
        nim_fasele = '‌'
        if not ignore_nim_fasele and nim_fasele in word:
            return self.handle_nim_fasele_words(word, pos)
        # pass ignore words and accept as correct informal word!
        if word in self.ignore_words and not word in self.mapper:
            return [(word, word)]
        formal_prefix_obj = self.prefix_obj(word)
        if formal_prefix_obj:
            return formal_prefix_obj
        iword = InformalWord(lemma=word, pos=pos)
        expanded_candidates = []
        candidates = self.parse_word(iword)
        #just verbs
        if any(c.is_verb for c in candidates):
            candidates = [c for c in candidates if c.is_verb]
        for cnd in candidates:
            expanded_candidates.extend(self.get_expand(cnd))
        word_iwords = []
        for ec in expanded_candidates:
            word_iwords.extend(self.to_formals(ec))
        if any(f[1] and (f[1].is_mapper or f[1].is_verb) for f in word_iwords if f[1] is not None):
            word_iwords = [f for f in word_iwords if f[1] and (f[1].is_mapper or f[1].is_verb)]
        # else:
        word_lemmas_set = [(w, iword.lemma) for w, iword in word_iwords if iword is not None]
        word_lemmas_set = set(word_lemmas_set)
        out = self.filter_results(word_lemmas_set)
        # if type(out) == str:
        #     out = [out]
        # out = set(out)
        return out

if __name__ == '__main__':
    transformer = OneShotTransformer(None, None, None)
    candidates =  transformer.match_postfixs('کارامم')
    print(candidates)

