import math
from package.generate_ngram import generate_ngram, preprocess_tokens
import package.hyperparameter_parser as hp

LOG_PROB_WHEN_NOT_FOUND = math.log( hp.parse_item_in_unk('fraction_of_oov'),2)

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
    scores = [calc_score(sentence, lm[i], i+1) for i in range(eval_hp['max_ngrams_in_language_models'])]
    interpolated_score = sum([scores[i]*eval_hp['interpolation_weight'][i] for i in range(len(eval_hp['interpolation_weight']))])
    return interpolated_score

def update_eval_hyperparameters(hps_to_update):
    eval_hyperparameters = hp.parse_evaluation_section()
    eval_hyperparameters.update(hps_to_update)
    return eval_hyperparameters
