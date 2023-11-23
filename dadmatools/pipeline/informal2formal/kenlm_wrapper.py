
class Kelm_Wrapper:
    def __init__(self, model):
        self.model = model
    def get_best_candidate_word(self, default_phrases, candidate_phrases, index):
        candidate_texts = [' '.join(default_phrases[:index]) + ' ' + cnd + ' ' + ' '.join(default_phrases[index+1:]) for cnd in candidate_phrases]
        scores = list(map(self.model.score, candidate_texts))
        return scores.index(max(scores))


    def get_best_ongram_phrases(self, candidates_list):
        bests = []
        for candidate_phrase in candidates_list:
            scores = list(map(self.model.score, candidate_phrase))
            best_phrase = candidate_phrase[scores.index(max(scores))]
            bests.append(best_phrase)
        return bests


    def get_best(self, candidates_list):
        bests = []
        default_phrases = self.get_best_ongram_phrases(candidates_list)
        # print(default_phrases)
        for index in range(len(candidates_list)):
            if len(candidates_list[index]) > 1:
                best_phrase_index = self.get_best_candidate_word(default_phrases, candidates_list[index], index)
                bests.append(candidates_list[index][best_phrase_index])
            else:
                bests.append(candidates_list[index][0])
        return ' '.join(bests)

