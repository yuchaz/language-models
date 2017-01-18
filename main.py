from package import load
from package.train import train_language_models
from package.evaluation import calc_perplexity, calc_interpolated_perplexity
from package.oov_words_handling import replace_low_freq_words
from package.hyperparameter_parser import parse_whole_updated_hps
import time
import sys
import json

def run_experiment(hps_to_update={}):
    training_corpus = load.load_traing_corpus()
    language_models = train_language_models(training_corpus,hps_to_update)
    dev_corpus_orig = load.load_dev_corpus()
    dev_corpus = replace_low_freq_words(dev_corpus_orig,hps_to_update)
    perplexity_orig = [calc_perplexity(dev_corpus, language_models[i], i+1) for i in range(3)]
    perplexity_interpolated = calc_interpolated_perplexity(dev_corpus, language_models, hps_to_update)
    print '{}\t{}\t{}\t{}\t{}'.format(
            json.dumps(hps_to_update), perplexity_orig[0],
            perplexity_orig[1], perplexity_orig[2], perplexity_interpolated
        )


def main():
    start = time.time()
    try:
        print 'hyperparameter\tperplexity_uni\tperplexity_bi\tperplexity_tri\tperplexity_inter'
        hps_to_update = [{}] if len(sys.argv) == 1 else parse_whole_updated_hps(sys.argv[1])
        for hps in hps_to_update:
            run_experiment(hps)
        print 'Time spent is {} seconds'.format(time.time()-start)

    except RuntimeError as error:
        print error


if __name__ == '__main__':
    main()
