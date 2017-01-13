from package.load import load_corpus
from package.train import train_language_models
from package.evaluation import calc_perplexity

CORPUS_PATH = './prob1_brown_full/'
TRAINING_CORPUS = CORPUS_PATH+'brown.train.txt'
DEV_CORPUS = CORPUS_PATH+'brown.dev.txt'
TEST_CORPUS = CORPUS_PATH+'brown.test.txt'

def main():
    try:
        training_corpus = load_corpus(TRAINING_CORPUS)
        unigram_p, bigram_p, trigram_p = train_language_models(training_corpus)
        # print unigram_p
        dev_corpus = load_corpus(DEV_CORPUS)
        perplexity = calc_perplexity(dev_corpus, bigram_p, 2)
        print perplexity

    except RuntimeError as error:
        print error


if __name__ == '__main__':
    main()
