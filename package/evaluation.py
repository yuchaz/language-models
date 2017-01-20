import math
from package.generate_ngram import generate_ngram, preprocess_tokens
import package.hyperparameter_parser as hp

LOG_PROB_WHEN_NOT_FOUND = hp.parse_item_in_constant('log_prob_when_not_found')
COUNT_WHEN_NOT_FOUND = hp.parse_item_in_constant('count_when_not_found')
KEYS_SHOULDNOT_BE_UPDATED = [
    'max_ngrams_in_language_models',
    'count_when_not_found',
    'log_prob_when_not_found'
]

def calc_score(sentence, ngram_lm, n, k=0):
    tokens = preprocess_tokens(sentence,n)
    ngrams_token = generate_ngram(tokens, n)
    return sum([calc_probability(token,ngram_lm, n, k) for token in ngrams_token])

def calc_probability(token, ngram_lm, n, k=0):
    numerator_count, denominator_count = get_numerator_and_denominator(token, ngram_lm, n,k)
    if numerator_count==0 or denominator_count==0:
        return LOG_PROB_WHEN_NOT_FOUND
    return math.log(float(numerator_count)/denominator_count,2)

def get_numerator_and_denominator(token,ngram_lm,n,k):
    vocab_size = ngram_lm[2]
    numerator_count = ngram_lm[0].get(token, COUNT_WHEN_NOT_FOUND)
    denominator_count = ngram_lm[1] if n==1 else ngram_lm[1].get(tuple(token[:n-1]), COUNT_WHEN_NOT_FOUND)
    return numerator_count+k, denominator_count+vocab_size*k

def calc_perplexity(corpus, ngram_lm, n, hps_to_update={}):
    eval_hp = update_eval_hyperparameters(hps_to_update)
    return general_perplexity_calculator(corpus,ngram_lm,n,eval_hp.get('k_to_add'))

def calc_perplexity_without_k(corpus,ngram_lm, n):
    return general_perplexity_calculator(corpus,ngram_lm,n)

def general_perplexity_calculator(corpus,ngram_lm,n,k=0):
    M = 0
    perplexity = .0
    for sentence in corpus:
        score = calc_score(sentence, ngram_lm, n, k)
        M += (len(sentence)+1)
        perplexity += score

    perplexity /= M
    perplexity = 2 ** (-1*perplexity)

    return perplexity

def calc_interpolated_perplexity(corpus, lm, hps_to_update={}):
    eval_hp = update_eval_hyperparameters(hps_to_update)
    M = 0
    perplexity = .0
    for sentence in corpus:
        score = interpolation_score(sentence, lm, eval_hp)
        M += (len(sentence)+1)
        perplexity += score

    perplexity /= M
    perplexity = 2** (-1*perplexity)
    return perplexity

def interpolation_score(sentence, lm, eval_hp):
    score = sum([2**calc_score(sentence, lm[i], i+1, eval_hp.get('k_to_add'))*eval_hp['interpolation_weight'][i] for i in range(len(eval_hp['interpolation_weight']))])
    if score == 0:
        return LOG_PROB_WHEN_NOT_FOUND
    return math.log(score,2)

def update_eval_hyperparameters(hps_to_update):
    if any([k in hps_to_update for k in KEYS_SHOULDNOT_BE_UPDATED]):
        print 'WARNING: some keys might not be updated since constant section should remain intact'
    eval_hyperparameters = hp.parse_evaluation_section()
    eval_hyperparameters.update(hps_to_update)
    return eval_hyperparameters
