from package.load import load_corpus
from package.train import train_language_models

CORPUS_PATH = './prob1_brown_full/'
TRAINING_CORPUS = CORPUS_PATH+'brown.train.txt'
DEV_CORPUS = CORPUS_PATH+'brown.dev.txt'
TEST_CORPUS = CORPUS_PATH+'brown.test.txt'

def main():
    training_corpus = load_corpus(TRAINING_CORPUS)
    unigram_p, bigram_p, trigram_p = train_language_models(training_corpus)
    print unigram_p

if __name__ == '__main__':
    main()
