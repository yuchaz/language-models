from package import load
from package.train import train_language_models
from package.evaluation import calc_perplexity, calc_interpolated_perplexity
from package.oov_words_handling import replace_low_freq_words
import time

def main():
    start = time.time()
    try:
        training_corpus = load.load_traing_corpus()
        language_models = train_language_models(training_corpus)
        dev_corpus_orig = load.load_dev_corpus()
        dev_corpus = replace_low_freq_words(dev_corpus_orig)
        perplexity_orig = [calc_perplexity(dev_corpus, language_models[i], i+1) for i in range(3)]
        perplexity_interpolated = calc_interpolated_perplexity(dev_corpus, language_models)
        print '\nPerplexity without interpolation for uni, bi, and trigram models are {}, {} and {}, respectively'.format(
            perplexity_orig[0], perplexity_orig[1], perplexity_orig[2]
        )
        print '\nPerplexity with interpolation is {}'.format(perplexity_interpolated)
        print 'Time spent is {} seconds'.format(time.time()-start)

    except RuntimeError as error:
        print error


if __name__ == '__main__':
    main()
