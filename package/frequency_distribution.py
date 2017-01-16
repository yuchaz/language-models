import nltk

def calc_freq_dist(lm_list):
    return dict(nltk.FreqDist(lm_list))
