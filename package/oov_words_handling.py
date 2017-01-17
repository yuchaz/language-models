import re
from package.frequency_distribution import calc_freq_dist
import random

STOP_SYMBOL = 'STOP'
UNK_SYMBOL = '<UNK>'
UNK_THRESHOLD = 5

TOTAL_UNK_WORDS_PROB = 0.01
OOV_AS_UNK_PROB = 0.7

def replace_low_freq_words(corpus):
    unigram_list = []
    for sentence in corpus:
        tokens = sentence.split(' ')+[STOP_SYMBOL]
        unigram_list.extend(tokens)
    unigram_freq_dist = calc_freq_dist(unigram_list)
    vocabulary_size = len(unigram_list)
    # words_to_replace = [k for k,v in unigram_freq_dist.items() if v <= UNK_THRESHOLD]
    words_to_replace = generate_replce_scheme(unigram_freq_dist, vocabulary_size)
    replaced_regex = re.compile('|'.join(map(re.escape, words_to_replace)))
    replaced_corpus = [replaced_regex.sub(UNK_SYMBOL,sentence) for sentence in corpus]
    return replaced_corpus

def generate_replce_scheme(unigram_freq_dist, vocabulary_size):
    unk_count = 0
    words_to_replace = []
    for k,v in unigram_freq_dist.items():
        if v <= UNK_THRESHOLD and \
           unk_count <= TOTAL_UNK_WORDS_PROB*vocabulary_size and \
           random.uniform(0,1) < OOV_AS_UNK_PROB:
            words_to_replace.append(k)
    return words_to_replace

def add_k_smoothing(numer, demon, vocab_size, k):
    return (numer+k)/(demon+k*vocab_size)
