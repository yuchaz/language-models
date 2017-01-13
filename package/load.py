NEWLINE_CHAR = '\n'
CORPUS_PATH = './prob1_brown_full/'
TEST_CORPUS = CORPUS_PATH+'brown.test.txt'

def load_corpus(path):
    if path == TEST_CORPUS:
        raise RuntimeError('You should not use test corpus at this time.')

    with open(path,'r') as infile:
        corpus = [line.rstrip(NEWLINE_CHAR) for line in infile]
    infile.close()
    return corpus
