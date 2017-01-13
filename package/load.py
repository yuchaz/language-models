NEWLINE_CHAR = '\n'

def load_corpus(path):
    with open(path,'r') as infile:
        corpus = [line.rstrip(NEWLINE_CHAR) for line in infile]
    infile.close()
    return corpus
