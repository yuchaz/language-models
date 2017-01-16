import nltk
import numpy
import math

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MAXIMUM_TRAIN_N_GRAM = 3

def train_language_models(corpus):
    ngram_result = []
    for n in range(MAXIMUM_TRAIN_N_GRAM):
        n_gram_freq_dist,n_1_gram_freq_dist = train_ngram_model(corpus,n+1)
        if n == 0:
            ngram_result.append({k: math.log( float(v)/n_1_gram_freq_dist, 2) for k,v in n_gram_freq_dist.items()})
        else:
            ngram_result.append({k: math.log( float(v)/n_1_gram_freq_dist[tuple(k[:n])], 2) for k,v in n_gram_freq_dist.items()})
    return ngram_result[0],ngram_result[1],ngram_result[2]

def train_ngram_model(corpus, n):
    n_gram_list = []
    n_1_gram_list = []
    for sentence in corpus:
        tokens = sentence.split(' ')
        tokens = [START_SYMBOL for i in range(n-1)]+tokens+[STOP_SYMBOL]
        n_gram_model = nltk.ngrams(tokens,n)
        if n != 1:
            n_1_gram_model = nltk.ngrams(tokens,n-1)
            n_1_gram_list.extend(n_1_gram_model)

        n_gram_list.extend(n_gram_model)

    if n != 1:
        return dict(nltk.FreqDist(n_gram_list)), dict(nltk.FreqDist(n_1_gram_list))
    else:
        return dict(nltk.FreqDist(n_gram_list)), len(n_gram_list)
