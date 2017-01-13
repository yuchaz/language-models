import nltk
import numpy
import math

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'

def train_language_models(corpus):
    unigrams_full_list = []
    unigrams_list_no_start = []
    bigrams_full_list = []
    bigrams_list_2_start = []
    trigrams_full_list = []
    for sentence in corpus:
        tokens = sentence.split(' ')
        tokens.append(STOP_SYMBOL)
        unigrams_list_no_start.extend(tokens)
        tokens.insert(0,START_SYMBOL)
        unigrams_full_list.extend(tokens)
        bigrams_full_list.extend(list(nltk.bigrams(tokens)))

        tokens.insert(0,START_SYMBOL)
        bigrams_list_2_start.extend(list(nltk.bigrams(tokens)))
        trigrams_full_list.extend(list(nltk.trigrams(tokens)))

    uni_fdist = dict(nltk.FreqDist(unigrams_full_list))
    uni_fdist_no_start = dict(nltk.FreqDist(unigrams_list_no_start))
    bi_fdist = dict(nltk.FreqDist(bigrams_full_list))
    bi_fdist_2_start = dict(nltk.FreqDist(bigrams_list_2_start))
    tri_fdist = dict(nltk.FreqDist(trigrams_full_list))

    unigram_p = {(k,): math.log( float(v)/len(unigrams_list_no_start), 2) for k,v in uni_fdist_no_start.items()}
    bigram_p = {k: math.log( float(v)/uni_fdist[k[0]], 2) for k,v in bi_fdist.items()}
    trigram_p = {k: math.log( float(v)/bi_fdist_2_start[(k[0],k[1])], 2) for k,v in tri_fdist.items()}

    return unigram_p, bigram_p, trigram_p
