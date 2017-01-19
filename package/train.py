import numpy
import math
from package.generate_ngram import generate_ngram, preprocess_tokens
from package.frequency_distribution import calc_freq_dist
from package.oov_words_handling import replace_low_freq_words

MAXIMUM_TRAIN_N_GRAM = 3

def train_language_models(corpus, hps_to_update={}):
    new_corpus = replace_low_freq_words(corpus, hps_to_update)
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
    vocabulary_size = get_vocabulary_size(corpus)
    if n != 1:
        n_1_gram_freq_dist = calc_freq_dist(n_1_gram_list)
        # return {k: math.log( float(v)/n_1_gram_freq_dist[tuple(k[:n-1])], 2) for k,v in n_gram_freq_dist.items()}
        return n_gram_freq_dist, n_1_gram_freq_dist, vocabulary_size
    else:
        # return {k: math.log( float(v)/len(n_gram_list), 2) for k,v in n_gram_freq_dist.items()}
        return n_gram_freq_dist, len(n_gram_list), vocabulary_size

def get_vocabulary_size(corpus):
    unigram_list = [token for sentence in corpus for token in preprocess_tokens(sentence,1)]
    unigram_freq_dist = calc_freq_dist(unigram_list)
    return len(unigram_freq_dist)
