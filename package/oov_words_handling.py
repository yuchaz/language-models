import re
import random
from package.frequency_distribution import calc_freq_dist
from package.generate_ngram import preprocess_tokens, replace_with_UNK
import package.hyperparameter_parser as hp

UNK_THRESHOLD = hp.parse_item_in_unk('unk_threshold')
FRACTION_OF_OOV = hp.parse_item_in_unk('fraction_of_oov')
CONVERT_OOV_TO_UNK_PROB = hp.parse_item_in_unk('convert_oov_to_unk_prob')

def replace_low_freq_words(corpus):
    unigram_list = [token for sentence in corpus for token in preprocess_tokens(sentence,1)]
    unigram_freq_dist = calc_freq_dist(unigram_list)
    vocabulary_size = len(unigram_list)
    words_to_replace = generate_replce_scheme(unigram_freq_dist, vocabulary_size)
    return replace_with_UNK(corpus, words_to_replace)

def generate_replce_scheme(unigram_freq_dist, vocabulary_size):
    unk_count = 0
    words_to_replace = []
    for k,v in unigram_freq_dist.items():
        if v <= UNK_THRESHOLD and \
           unk_count <= FRACTION_OF_OOV*vocabulary_size and \
           random.uniform(0,1) < CONVERT_OOV_TO_UNK_PROB:
            words_to_replace.append(k)
    return words_to_replace

def add_k_smoothing(numer, demon, vocab_size, k):
    return (numer+k)/(demon+k*vocab_size)
