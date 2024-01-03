import re
from enum import Enum
from dadmatools.normalizer import Normalizer
import pandas as pd


Formality = Enum('Formality', 'formal informal')
VerbTime = Enum('VerbTime', 'past present future')
Person = Enum('Person', 'Man To An Ma Shoma Anha')
Number = Enum('Number', 'Mofrad Jam')
class Verb:
    def __init__(self, root, formality, time, pp, person, number):
        self.root = root
        self.formality = formality
        self.time = time
        self.person = person
        self.number = number
        self.pp = pp

class VerbHandler():
    def __init__(self, csv_verb_addr, csv_irregular_verbs_mapper):
        self.posfix_mapper = {'ه': 'د', 'ن': 'ند', 'ین': 'ید'}
        self.objective_pr_mapper = {'شون':'شان', 'تون':'تان', 'مون':'مان'}
        self.init_mapper = {'کنه': 'بکنه', 'کنم':'بکنم', 'کنی':'بکنی', 'کنیم': 'بکنیم', 'کنین':'بکنین', 'کنید':'بکنید', 'کنن':'بکنن', 'کنند':'بکنند'}
        self.out_mapper = {'می‌ایی': 'می‌آیی'}
        self.init_mapper.update({'شم':'بشم', 'شی':'بشی', 'شن':'بشن', 'شین':'بشین' ,'شه':'بشه', 'شیم': 'بشیم'})
        self.bons = self.load_bons(csv_verb_addr)
        self.irregular_verbs = self.load_irregular_mapper(csv_irregular_verbs_mapper)
        self.informal_past_bons = self.get_bons(type=Formality.informal, time=VerbTime.past)
        self.informal_present_bons = self.get_bons(type=Formality.informal, time=VerbTime.present)

        self.formal_past_bons = self.get_bons(type=Formality.formal, time=VerbTime.past)
        self.formal_present_bons =self.get_bons(type=Formality.formal, time=VerbTime.present) + ['هست']
        self.all_past_bons = self.formal_past_bons + self.informal_past_bons
        self.all_present_bons = self.formal_present_bons + self.informal_present_bons
        self.verb_mapper = {b:{'formal':self.bons[b]['formal']} for b in self.bons if self.bons[b]['type'] == Formality.informal}
        self.solve_alef_issue()
        self.compile_patterns()


    def load_irregular_mapper(self, csv_addr):
        df = pd.read_csv(csv_addr)
        mapper = {informal: formal for _, (informal, formal) in df.iterrows()}
        return mapper

    def load_bons(self, csv_addr):
        normalizer = Normalizer()
        df = pd.read_csv(csv_addr)
        df = df.fillna('')
        bons = {}
        for i, row in df.iterrows():
            if row[2]:
                row[2] = normalizer.normalize(row[2])
                bons[row[2]] = {'type': Formality.formal, 'time': VerbTime.past}
            if row[3]:
                row[3] = normalizer.normalize(row[3])
                bons[row[3]] = {'type': Formality.formal, 'time': VerbTime.present}
            if row[10]:
                bs = row[10].split()
                for b in bs:
                    bons[b] = {'type': Formality.informal, 'time': VerbTime.past, 'formal': row[2]}
            if row[11]:
                bs = row[11].split()
                for b in bs:
                    bons[b] = {'type': Formality.informal, 'time': VerbTime.present, 'formal': row[3]}
        return bons

    def get_bons(self, type, time):
        return [b for b in self.bons if self.bons[b]['type'] == type and self.bons[b]['time'] == time]

    def solve_alef_issue(self):
        replace_alef_y = lambda v : 'ی' + v[1:]
        replace_A_YA = lambda v : 'یا' + v[1:]
        informal_past_start_with_alef = list(map(replace_alef_y, [v for v in self.informal_past_bons if v.startswith('ا') and not v.startswith('ای')]))
        formal_past_start_with_alef = list(map(replace_alef_y, [v for v in self.formal_past_bons if v.startswith('ا') and not v.startswith('ای')]))
        informal_present_start_with_alef = list(map(replace_alef_y, [v for v in self.informal_present_bons if v.startswith('ا') and not v.startswith('ای')]))
        formal_present_start_with_alef = list(map(replace_alef_y, [v for v in self.formal_present_bons if v.startswith('ا') and not v.startswith('ای')]))
        self.alef_mapper = {}
        self.informal_past_start_with_alef = informal_past_start_with_alef + list(
            map(replace_A_YA, [v for v in self.informal_past_bons if v.startswith('آ')]))
        self.informal_present_start_with_alef = informal_present_start_with_alef + list(
            map(replace_A_YA, [v for v in self.informal_present_bons if v.startswith('آ')]))
        self.formal_past_start_with_alef = formal_past_start_with_alef + list(
            map(replace_A_YA, [v for v in self.formal_past_bons if v.startswith('آ')]))
        self.formal_present_start_with_alef = formal_present_start_with_alef + list(
            map(replace_A_YA, [v for v in self.formal_present_bons if v.startswith('آ')]))
        for verb in self.informal_past_start_with_alef + self.informal_present_start_with_alef + self.formal_past_start_with_alef + self.formal_present_start_with_alef:
            if verb[:2] == 'یا':
                origin = 'آ' + verb[2:]
            else:
                origin = 'ا' + verb[1:]
            self.alef_mapper[verb] = origin
        self.alef_mapper['یای'] = 'آی'
        remove_a_hat = lambda w: w.replace('آ', 'ا')
        self.formal_past_bons = list(
            filter(lambda w: w != '', map(remove_a_hat, self.formal_past_bons + self.formal_past_start_with_alef)))
        self.formal_present_bons = list(map(remove_a_hat, self.formal_present_bons + self.formal_present_start_with_alef)) + [
            'یای'] + ['آی']
        self.informal_past_bons = list(
            filter(lambda w: w != '', map(remove_a_hat, self.informal_past_bons + self.informal_past_start_with_alef)))
        self.informal_present_bons = list(
            map(remove_a_hat, self.informal_present_bons + self.informal_present_start_with_alef)) + [
                                       'یای'] + ['آی']
        # sorted by length
        self.formal_present_bons = sorted(self.formal_present_bons, key=lambda w: -len(w))
        self.formal_past_bons = sorted(self.formal_past_bons, key=lambda w: -len(w))
        self.informal_present_bons = sorted(self.informal_present_bons, key=lambda w: -len(w))
        self.informal_past_bons = sorted(self.informal_past_bons, key=lambda w: -len(w))
        verb_v_keys = [word for word in self.verb_mapper if 'آ' in word]
        alef_verb_v_keys = [word for word in self.alef_mapper if 'آ' in word]
        for v in verb_v_keys:
            self.verb_mapper[v.replace('آ', 'ا')] = self.verb_mapper[v]
        for v in alef_verb_v_keys:
            self.alef_mapper[v.replace('آ', 'ا')] = self.alef_mapper[v]


    def compile_patterns(self):
            ME_r = '|'.join(['می','می‌'])
            B_r = 'ب'
            not_r = 'ن'
            past_ends = ['م', 'ی', 'ه', 'یم', 'ین', 'ید', 'ند', '', 'ن']
            present_ends = ['م', 'ی', 'ه', 'یم', 'ین', 'ن', 'ید', 'ند', 'د', '']
            naghli_ends = ['ه‌ام', 'ه‌ای', 'ه', 'ه‌ایم', 'ه‌اید', 'ه‌اند']
            objective_pronouns = ['م', 'ت', 'ش', 'مون', 'تون', 'شون']

            informal_past_r = '|'.join(self.informal_past_bons)
            formal_past_r = '|'.join(self.formal_past_bons)
            informal_present_r = '|'.join(self.informal_present_bons)
            formal_present_r = '|'.join(self.formal_present_bons)
            verb_postfix_past_r = '|'.join(past_ends)
            verb_postfix__present_r = '|'.join(present_ends)
            objective_pronouns_r = '|'.join(objective_pronouns)
            naghli_ends_r = '|'.join(naghli_ends)
            """
            #گذشته‌ی ساده
            # r1 =  past_r + verb_postfix_r + objectiveـpronouns_r
            #گذشته‌ی ناتمام
            # r2  = '(' + ME + ')'+ past_r +verb_postfix_r + objectiveـpronouns_r
    
            #گذشته‌ی استمراری
            # r3 =  '(' + DASHT + ')'+ past_r +  verb_postfix_r +objectiveـpronouns_r
    
            #گذشته‌ی نقلی
            # r4 = past_r + '(' + '|'.join(naghli_ends) + ')' +objectiveـpronouns_r
    
            #گذشته‌ی پیشین
            # r5 = past_r + verb_postfix_r + '(' + BUD + ')' + verb_postfix_r + objectiveـpronouns_r
    
            #حال ساده
            # r6 = present_r + verb_postfix_r
    
           #حال ناتمام
            # r7 =  '(' + ME + ')'+ present_r +  verb_postfix_r + objectiveـpronouns_r
    
            #حال استمراری
            # r8 = '( ' + DAR + ')'+ verb_postfix_r + '(' + ME + ')' + present_r + verb_postfix_r+ objectiveـpronouns_r
    
            #آینده‌ی ساده
            # r9 = '( ' + KHAH + ')'+ verb_postfix_r + present_r +objectiveـpronouns_r
    
            #التزامی - گذشته
            # r10 = present_r + '(ه)'+  '(' + BASH +  ')' + verb_postfix_r + objectiveـpronouns_r
    
            #التزامی - حال
            # r11 = '(ب)' + present_r + verb_postfix_r +objectiveـpronouns_r
            """
            #
            # + : fealhaye rasmi + pasvan informal , hale sade baraye bazi fela ( hast, kon)
            # formal
            formal_present_pattern_b = '({})({})({})?({})?$'.format(B_r, formal_present_r, verb_postfix__present_r, objective_pronouns_r)
            formal_present_pattern_n_me = '({})?({})({})({})?({})?$'.format(not_r, ME_r, formal_present_r, verb_postfix__present_r, objective_pronouns_r)
            formal_present_pattern_n = '({})?({})({})?({})?$'.format(not_r, formal_present_r, verb_postfix__present_r, objective_pronouns_r)
            formal_past_pattern = '({})?({})?({})({}|{})({})?$'.format(not_r, ME_r, formal_past_r, naghli_ends_r,
                                                                        verb_postfix_past_r, objective_pronouns_r)
            self.formal_past_verb_pattern = re.compile(formal_past_pattern)
            self.formal_present_verb_pattern_b = re.compile(formal_present_pattern_b)
            self.formal_present_verb_pattern_n_me = re.compile(formal_present_pattern_n_me)
            self.formal_present_verb_pattern_n = re.compile(formal_present_pattern_n)

            #informal
            informal_present_pattern_b = '({})({})({})?({})?$'.format(B_r, informal_present_r, verb_postfix__present_r,
                                                                     objective_pronouns_r)
            informal_present_pattern_n_me = '({})?({})({})({})?({})?$'.format(not_r, ME_r, informal_present_r,
                                                                             verb_postfix__present_r, objective_pronouns_r)
            informal_present_pattern_n = '({})?({})({})?({})?$'.format(not_r, informal_present_r, verb_postfix__present_r,
                                                                      objective_pronouns_r)
            informal_past_pattern = '({})?({})?({})({}|{})({})?$'.format(not_r, ME_r, informal_past_r, naghli_ends_r,
                                                                        verb_postfix_past_r, objective_pronouns_r)
            self.informal_past_verb_pattern = re.compile(informal_past_pattern)
            self.informal_present_verb_pattern_b = re.compile(informal_present_pattern_b)
            self.informal_present_verb_pattern_n_me = re.compile(informal_present_pattern_n_me)
            self.informal_present_verb_pattern_n = re.compile(informal_present_pattern_n)


    def parse(self, token):
        outputs = []

        match_dict_formal = {'tense': '', 'root': '', 'neg': '', 'postfix': '', 'not_r': '', 'op': '', 'b': '', 'me': '', 'naghli':''}
        match_dict_informal = {'tense': '', 'root': '', 'neg': '', 'postfix': '', 'not_r': '', 'op': '', 'b': '', 'me': '', 'naghli':''}
        formal_past_match = self.formal_past_verb_pattern.match(token)
        informal_past_match = self.informal_past_verb_pattern.match(token)
        formal_present_match_b = self.formal_present_verb_pattern_b.match(token)
        informal_present_match_b = self.informal_present_verb_pattern_b.match(token)
        formal_present_match_n_me = self.formal_present_verb_pattern_n_me.match(token)
        informal_present_match_n_me = self.informal_present_verb_pattern_n_me.match(token)
        formal_present_match_n = self.formal_present_verb_pattern_n.match(token)
        informal_present_match_n = self.informal_present_verb_pattern_n.match(token)
        present_group_to_dict_b = lambda g: {k:g[i] for i,k in enumerate(['b', 'root', 'postfix', 'op'])}
        present_group_to_dict_n_me = lambda g: {k:g[i] for i,k in enumerate(['neg', 'me','root', 'postfix','op'])}
        present_group_to_dict_n = lambda g: {k:g[i] for i,k in enumerate(['neg','root', 'postfix','op'])}
        past_group_to_dict = lambda g: {k:g[i] for i,k in enumerate(['neg', 'me', 'root', 'postfix', 'op'])}
        formal_match = formal_past_match or formal_present_match_b or formal_present_match_n_me or formal_present_match_n
        informal_match = informal_past_match or informal_present_match_b or informal_present_match_n_me or informal_present_match_n
        if formal_match:
            if formal_past_match:
                match_dict_formal = past_group_to_dict(formal_past_match.groups())
                match_dict_formal['tense'] = 'past'
            else:
                if formal_present_match_b:
                    match_dict_formal = present_group_to_dict_b(formal_present_match_b.groups())
                elif formal_present_match_n_me:
                    match_dict_formal = present_group_to_dict_n_me(formal_present_match_n_me.groups())
                elif formal_present_match_n:
                    match_dict_formal = present_group_to_dict_n(formal_present_match_n.groups())
                match_dict_formal['tense'] = 'present'
            outputs.append(match_dict_formal)
        if informal_match:
            if informal_past_match:
                match_dict_informal = past_group_to_dict(informal_past_match.groups())
                match_dict_informal['tense'] = 'past'
            else:
                if informal_present_match_b:
                    match_dict_informal = present_group_to_dict_b(informal_present_match_b.groups())
                elif informal_present_match_n_me:
                    match_dict_informal = present_group_to_dict_n_me(informal_present_match_n_me.groups())
                elif informal_present_match_n:
                    match_dict_informal = present_group_to_dict_n(informal_present_match_n.groups())
                match_dict_informal['tense'] = 'present'
            outputs.append(match_dict_informal)
        for match_dict in outputs:
            for key,val in match_dict.items():
                if val is None:
                    match_dict[key] = ''
            # print(match_dict)
        return outputs

    def formal_concatenate(self, match_dict, should_smooth):
        out_dict = {'بیای': 'بیا', 'نیای': 'نیا'}
        if match_dict['root'] == 'است' and match_dict['neg'] != '':
            return 'نیست' + match_dict['postfix']
        if self.if_simple_present(match_dict) or self.if_only_me(match_dict):
            return None
        if should_smooth:
            if match_dict['prefix'] != '' and match_dict['prefix'][0] == 'م':
                pass
            else:
                match_dict['root'] = 'یا' + match_dict['root'][1:]
            # if len(match_dict['prefix']) == 3:
            #     match_dict['prefix'] = 'می'
        if match_dict['prefix'] == 'ب' and match_dict['root'] and match_dict['root'][0] == 'ا':
            match_dict['root'] = 'ی' + match_dict['root'][1:]
        out = match_dict['neg'] + match_dict['prefix'] + match_dict['root'] + match_dict['postfix'] + match_dict['op']
        if out in out_dict:
            out = out_dict[out]

        return out

    def _set_match_dict_prefix(self, match_dict):
        match_dict['prefix'] = ''
        if 'me' in match_dict and match_dict['me'] != '':
            if len(match_dict['me']) < 3:
                match_dict['me'] = 'می‌'
            match_dict['prefix'] = match_dict['me']
        elif 'b' in match_dict and match_dict['b'] != '':
            match_dict['prefix'] = match_dict['b']
        return match_dict

    def if_simple_present(self, match_dict):
        if match_dict['root'] != '' and match_dict['tense'] == 'present' and match_dict['prefix'] == '' and match_dict['neg'] == '':
            if match_dict['root'] not in ['کن', 'هست', 'است', 'دار', 'نیست', 'باش']:
                return True
        return False

    def if_only_me(self, match_dict):
        if match_dict['root'] != '' and match_dict['tense'] == 'present' and match_dict['prefix'] !='' and match_dict['prefix'][0] == 'م' and match_dict['postfix'] == '':
            return True
        return False

    def is_masdar(self, match_dict):
        return  match_dict['root'] in self.all_past_bons and match_dict['me'] == '' and match_dict['postfix'] =='ن' and match_dict['op'] == ''

    def informal_to_formal(self, token):
        # irregular verbs checking
        if token in self.irregular_verbs:
            return [self.irregular_verbs[token]]
        if token in self.init_mapper:
            token = self.init_mapper[token]
        outputs = []
        if len(token) < 3:
            return None
        should_smooth = False
        all_match_dicts = self.parse(token)

        ### بدهدم
        #برد
        if len(all_match_dicts) == 2 :
            if all_match_dicts[1]['root'] in self.verb_mapper and self.verb_mapper[all_match_dicts[1]['root']]['formal'] == all_match_dicts[0]['root'] and all_match_dicts[1]['op'] != '':
                del all_match_dicts[1]
            elif all_match_dicts[1] == {'b': 'ب', 'root': 'ر', 'postfix': 'د', 'op': '', 'tense': 'present'}:
                del all_match_dicts[1]
        ##
        is_masdar = False
        for match_dict in all_match_dicts:
            if self.is_masdar(match_dict):
                is_masdar = True
            #نان بان
            if match_dict['root'] != '' and match_dict['root'][0] == 'ا' and 'me' not in match_dict and ('b' in match_dict or match_dict['neg'] == 'ن'):
                return None
            if match_dict['root'] != '':
                root = match_dict['root']
                objective_pr = match_dict['op']
                postfix = match_dict['postfix']
                if root in self.alef_mapper:
                    should_smooth = True
                    match_dict['root'] = self.alef_mapper[root]
                if match_dict['root'] in self.verb_mapper:
                    match_dict['root'] = self.verb_mapper[ match_dict['root']]['formal']
                if postfix in self.posfix_mapper:
                    match_dict['postfix'] = self.posfix_mapper[postfix]
                if match_dict['postfix'] == 'د' and match_dict['tense'] == 'past':
                    match_dict['postfix'] = 'ه'
                if objective_pr in self.objective_pr_mapper:
                    match_dict['op'] = self.objective_pr_mapper[objective_pr]
                match_dict['prefix'] = ''
                if 'neg' not in match_dict:
                    match_dict['neg'] = ''
                match_dict = self._set_match_dict_prefix(match_dict)
                formal_verb = self.formal_concatenate(match_dict, should_smooth)
                outputs.append(formal_verb)
        not_none_outpts = [o for o in outputs if o is not None]
        for index, item in enumerate(not_none_outpts):
            if item in self.out_mapper:
                not_none_outpts[index] = self.out_mapper[item]
        if not_none_outpts:
            # append bon
            if len(not_none_outpts) == 1 and is_masdar:
                masdar = not_none_outpts[0][:-2] + 'ن'
                not_none_outpts.append(masdar)
            return not_none_outpts
        return None