# import nltk
from collections import defaultdict

def calc_freq_dist(lm_list):
    dictionary = defaultdict(int)
    for lm in lm_list:
        dictionary[lm] += 1
    return dictionary
    # return {lm: lm_list.count(lm) for lm in lm_list}
    # return dict(nltk.FreqDist(lm_list))
