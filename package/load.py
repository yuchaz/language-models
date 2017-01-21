CORPUS_PATH = './prob1_brown_full/'
TRAINING_CORPUS = CORPUS_PATH+'brown.train.txt'
DEV_CORPUS = CORPUS_PATH+'brown.dev.txt'
TEST_CORPUS = CORPUS_PATH+'brown.test.txt'
NEWLINE_CHAR = '\n'

def load_corpus(path):
    with open(path,'r') as infile:
        corpus = [line.rstrip(NEWLINE_CHAR) for line in infile]
    infile.close()
    return corpus

def load_traing_corpus():
    return load_corpus(TRAINING_CORPUS)

def load_dev_corpus():
    return load_corpus(DEV_CORPUS)

def load_test_corpus():
    return load_corpus(TEST_CORPUS)
