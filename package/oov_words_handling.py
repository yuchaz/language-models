import re
import random
from package.frequency_distribution import calc_freq_dist
from package.generate_ngram import preprocess_tokens, replace_with_UNK
import package.hyperparameter_parser as hp

def replace_low_freq_words(corpus, hps_to_update={}):
    unk_hp = update_unk_hyperparameters(hps_to_update)
    unigram_list = [token for sentence in corpus for token in preprocess_tokens(sentence,1)]
    unigram_freq_dist = calc_freq_dist(unigram_list)
    vocabulary_size = len(unigram_freq_dist)
    words_to_replace = generate_replce_scheme(unigram_freq_dist, vocabulary_size, unk_hp)
    return replace_with_UNK(corpus, words_to_replace)

def update_unk_hyperparameters(hps_to_update):
    unk_hyperparameters = hp.parse_unk_section()
    unk_hyperparameters.update(hps_to_update)
    return unk_hyperparameters

def generate_replce_scheme(unigram_freq_dist, vocabulary_size, unk_hp):
    unk_count = 0
    words_to_replace = []
    for k,v in random.sample(unigram_freq_dist.items(), len(unigram_freq_dist)):
        if v <= unk_hp['unk_threshold'] and \
           unk_count <= unk_hp['fraction_of_oov']*vocabulary_size and \
           random.uniform(0,1) < unk_hp['convert_oov_to_unk_prob']:
            words_to_replace.append(k)
            unk_count += 1
    return words_to_replace
