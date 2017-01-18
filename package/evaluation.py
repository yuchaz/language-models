import math
from package.generate_ngram import generate_ngram, preprocess_tokens
import package.hyperparameter_parser as hp

#LOG_PROB_WHEN_NOT_FOUND = hp.parse_item_in_evaluation('log_prob_when_not_found')
LOG_PROB_WHEN_NOT_FOUND = math.log( hp.parse_item_in_unk('fraction_of_oov'),2)
#import pdb; pdb.set_trace()
MAX_NGRAMS_IN_LM = hp.parse_item_in_evaluation('max_ngrams_in_language_models')
INTERPOLATION_WEIGHT = hp.parse_item_in_evaluation('interpolation_weight')

def calc_score(sentence, ngram_lm, n):
    tokens = preprocess_tokens(sentence, n)
    ngrams_token = generate_ngram(tokens, n)

    score = .0
    for token in ngrams_token:
        score += ngram_lm.get(token, LOG_PROB_WHEN_NOT_FOUND)

    return score

def calc_perplexity(corpus, ngram_lm, n):
    M = 0
    perplexity = .0
    for sentence in corpus:
        score = calc_score(sentence, ngram_lm, n)
        M += (len(sentence)+1)
        perplexity += score

    perplexity /= M
    perplexity = 2 ** (-1*perplexity)

    return perplexity

def calc_interpolated_perplexity(corpus, lm):
    M = 0
    perplexity = .0
    for sentence in corpus:
        score = interpolation_score(sentence, lm)
        M += (len(sentence)+1)
        perplexity += score

    perplexity /= M
    perplexity = 2** (-1*perplexity)
    return perplexity

def interpolation_score(sentence, lm):
    scores = [calc_score(sentence, lm[i], i+1) for i in range(MAX_NGRAMS_IN_LM)]
    interpolated_score = sum([scores[i]*INTERPOLATION_WEIGHT[i] for i in range(len(INTERPOLATION_WEIGHT))])
    return interpolated_score
