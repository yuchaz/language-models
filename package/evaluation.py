import math
from package.generate_ngram import generate_ngram, preprocess_tokens
import package.hyperparameter_parser as hp

LOG_PROB_WHEN_NOT_FOUND = math.log( hp.parse_item_in_unk('fraction_of_oov'),2)
COUNT_WHEN_NOT_FOUND = 0

def calc_score(sentence, ngram_lm, n):
    tokens = preprocess_tokens(sentence, n)
    ngrams_token = generate_ngram(tokens, n)
    score = sum([calc_probability(token, ngram_lm, n) for token in ngrams_token])
    return score

def calc_probability(token, ngram_lm,n):
    numerator_count, denominator_count = get_numerator_and_denominator(token, ngram_lm, n)
    if numerator_count==0 or denominator_count==0:
        return -1000
    return math.log(float(numerator_count)/denominator_count,2)

def get_numerator_and_denominator(token,ngram_lm,n):
    numerator_count = ngram_lm[0].get(token, COUNT_WHEN_NOT_FOUND)
    denominator_count = ngram_lm[1] if n==1 else ngram_lm[1].get(tuple(token[:n-1]), COUNT_WHEN_NOT_FOUND)
    return numerator_count, denominator_count

def calc_add_k_score(sentence, ngram_lm, n, k):
    tokens = preprocess_tokens(sentence,n)
    ngrams_token = generate_ngram(tokens, n)
    return sum([calc_add_k_probability(token,ngram_lm, n, k) for token in ngrams_token])

def calc_add_k_probability(token, ngram_lm, n, k):
    numerator_count, denominator_count = get_add_k_numerator_and_denominator(token, ngram_lm, n,k)
    return math.log(float(numerator_count)/denominator_count,2)

def get_add_k_numerator_and_denominator(token,ngram_lm,n,k):
    vocab_size = ngram_lm[2]
    numerator_count, denominator_count = get_numerator_and_denominator(token, ngram_lm, n)
    return numerator_count+k, denominator_count+vocab_size*k

def calc_perplexity(corpus, ngram_lm, n, k=0):
    M = 0
    perplexity = .0
    for sentence in corpus:
        score = calc_score(sentence, ngram_lm, n) if k==0 else calc_add_k_score(sentence,ngram_lm,n,k)
        M += (len(sentence)+1)
        perplexity += score

    perplexity /= M
    perplexity = 2 ** (-1*perplexity)

    return perplexity

def calc_interpolated_perplexity(corpus, lm, k=0, hps_to_update={}):
    eval_hp = update_eval_hyperparameters(hps_to_update)
    M = 0
    perplexity = .0
    for sentence in corpus:
        score = interpolation_score(sentence, lm, k, eval_hp)
        M += (len(sentence)+1)
        perplexity += score

    perplexity /= M
    perplexity = 2** (-1*perplexity)
    return perplexity

def interpolation_score(sentence, lm, k, eval_hp):
    score = sum([2**calc_add_k_score(sentence, lm[i], i+1, k)*eval_hp['interpolation_weight'][i] for i in range(len(eval_hp['interpolation_weight']))])
    if score == 0:
        return -1000
    return math.log(score,2)

def update_eval_hyperparameters(hps_to_update):
    eval_hyperparameters = hp.parse_evaluation_section()
    eval_hyperparameters.update(hps_to_update)
    return eval_hyperparameters
