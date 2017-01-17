from package.load import load_corpus
from package.train import train_language_models
from package.evaluation import calc_perplexity
from package.oov_words_handling import replace_low_freq_words
import time

CORPUS_PATH = './prob1_brown_full/'
TRAINING_CORPUS = CORPUS_PATH+'brown.train.txt'
DEV_CORPUS = CORPUS_PATH+'brown.dev.txt'
TEST_CORPUS = CORPUS_PATH+'brown.test.txt'

def main():
    start = time.time()
    try:
        training_corpus = load_corpus(TRAINING_CORPUS)
        unigram_p, bigram_p, trigram_p = train_language_models(training_corpus)
        dev_corpus_orig = load_corpus(DEV_CORPUS)
        dev_corpus = replace_low_freq_words(dev_corpus_orig)

        perplexity_uni = calc_perplexity(dev_corpus, unigram_p, 1)
        perplexity_bi = calc_perplexity(dev_corpus, bigram_p, 2)
        perplexity_tri = calc_perplexity(dev_corpus, trigram_p, 3)
        print 'Perplexity for uni, bi, and trigram models are {}, {} and {}, respectively'.format(
            perplexity_uni, perplexity_bi, perplexity_tri
        )
        print 'Time spent is {} seconds'.format(time.time()-start)

    except RuntimeError as error:
        print error


if __name__ == '__main__':
    main()
