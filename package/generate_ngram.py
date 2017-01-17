def generate_ngram(tokens,n):
    return zip(*[tokens[i:] for i in range(n)])
