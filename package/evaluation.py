from package.generate_ngram import generate_ngram, preprocess_tokens

MINUS_INFINITY_SENTENCE_LOG_PROB = -1000
MAXIMUM_TRAIN_N_GRAM = 3
INTERPOLATION_WEIGHT = [.33,.33,.34]


def calc_score(sentence, ngram_lm, n):
    tokens = preprocess_tokens(sentence, n)
    ngrams_token = generate_ngram(tokens, n)

    score = .0
    for token in ngrams_token:
        score += ngram_lm.get(token, MINUS_INFINITY_SENTENCE_LOG_PROB)

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
    scores = [calc_score(sentence, lm[i], i+1) for i in range(MAXIMUM_TRAIN_N_GRAM)]
    interpolated_score = sum([scores[i]*INTERPOLATION_WEIGHT[i] for i in range(len(INTERPOLATION_WEIGHT))])
    return interpolated_score
