import numpy
import math
from package.generate_ngram import generate_ngram, preprocess_tokens
from package.frequency_distribution import calc_freq_dist
from package.oov_words_handling import replace_low_freq_words

MAXIMUM_TRAIN_N_GRAM = 3

def train_language_models(corpus):
    new_corpus = replace_low_freq_words(corpus)
    ngram_result = [train_ngram_model(new_corpus,n+1) for n in range(MAXIMUM_TRAIN_N_GRAM)]
    return ngram_result

def train_ngram_model(corpus, n):
    n_gram_list = []
    n_1_gram_list = []
    for sentence in corpus:
        tokens = preprocess_tokens(sentence, n)
        n_gram_model = generate_ngram(tokens,n)
        n_gram_list.extend(n_gram_model)
        if n != 1:
            n_1_gram_model = generate_ngram(tokens,n-1)
            n_1_gram_list.extend(n_1_gram_model)

    n_gram_freq_dist = calc_freq_dist(n_gram_list)
    if n != 1:
        n_1_gram_freq_dist = calc_freq_dist(n_1_gram_list)
        return {k: math.log( float(v)/n_1_gram_freq_dist[tuple(k[:n-1])], 2) for k,v in n_gram_freq_dist.items()}
    else:
        return {k: math.log( float(v)/len(n_gram_list), 2) for k,v in n_gram_freq_dist.items()}
