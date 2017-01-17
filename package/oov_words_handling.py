import re
from package.frequency_distribution import calc_freq_dist

STOP_SYMBOL = 'STOP'
UNK_SYMBOL = '<UNK>'
UNK_THRESHOLD = 5

def replace_low_freq_words(corpus):
    unigram_list = []
    for sentence in corpus:
        tokens = sentence.split(' ')+[STOP_SYMBOL]
        unigram_list.extend(tokens)
    unigram_freq_dist = calc_freq_dist(unigram_list)
    words_to_replace = [k for k,v in unigram_freq_dist.items() if v <= UNK_THRESHOLD]
    replaced_regex = re.compile('|'.join(map(re.escape, words_to_replace)))
    replaced_corpus = [replaced_regex.sub(UNK_SYMBOL,sentence) for sentence in corpus]
    return replaced_corpus
