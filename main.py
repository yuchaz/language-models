from package import load
from package.train import train_language_models
from package.evaluation import calc_perplexity, calc_interpolated_perplexity, calc_perplexity_without_k
from package.oov_words_handling import replace_low_freq_words
from package.hyperparameter_parser import parse_whole_updated_hps, parse_unk_section, parse_evaluation_section, parse_item_in_constant
import time
import sys, pdb, traceback
import json

MAX_N_GRAM = parse_item_in_constant('max_ngrams_in_language_models')

def run_experiment(hps_to_update={}):
    training_corpus_orig = load.load_traing_corpus()
    training_corpus = replace_low_freq_words(training_corpus_orig, hps_to_update)
    language_models = train_language_models(training_corpus,hps_to_update)
    dev_corpus_orig = load.load_dev_corpus()
    dev_corpus = replace_low_freq_words(dev_corpus_orig,hps_to_update)
    test_corpus_orig = load.load_test_corpus()
    test_corpus = replace_low_freq_words(test_corpus_orig, hps_to_update)

    perplexity_train_set = [calc_perplexity_without_k(training_corpus, language_models[i], i+1) for i in range(MAX_N_GRAM)]
    perplexity_dev_set = [calc_perplexity_without_k(dev_corpus, language_models[i], i+1) for i in range(MAX_N_GRAM)]
    perplexity_test_set = [calc_perplexity_without_k(test_corpus, language_models[i], i+1) for i in range(MAX_N_GRAM)]

    perplexity_add_k_on_train = [calc_perplexity(training_corpus, language_models[i], i+1, hps_to_update) for i in range(MAX_N_GRAM)]
    perplexity_add_k_on_dev = [calc_perplexity(dev_corpus, language_models[i], i+1, hps_to_update) for i in range(MAX_N_GRAM)]
    perplexity_interpolated_on_train = calc_interpolated_perplexity(training_corpus, language_models, hps_to_update)
    perplexity_interpolated_on_dev = calc_interpolated_perplexity(dev_corpus, language_models, hps_to_update)
    print '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            json.dumps(hps_to_update),
            perplexity_train_set[0], perplexity_train_set[1], perplexity_train_set[2],
            perplexity_dev_set[0], perplexity_dev_set[1], perplexity_dev_set[2],
            perplexity_test_set[0], perplexity_test_set[1], perplexity_test_set[2],
            perplexity_add_k_on_train[0], perplexity_add_k_on_train[1], perplexity_add_k_on_train[2],
            perplexity_add_k_on_dev[0], perplexity_add_k_on_dev[1], perplexity_add_k_on_dev[2],
            perplexity_interpolated_on_train, perplexity_interpolated_on_dev
        )


def main():
    start = time.time()
    try:
        print json.dumps(parse_unk_section())
        print json.dumps(parse_evaluation_section())
        print 'hyperparameter\tperplexity_train_uni\tperplexity_train_bi\tperplexity_train_tri\tperplexity_dev_uni\tperplexity_dev_bi\tperplexity_dev_tri\tperplexity_test_uni\tperplexity_test_bi\tperplexity_test_tri\tperplexity_k_train_uni\tperplexity_k_train_bi\tperplexity_k_train_tri\tperplexity_k_dev_uni\tperplexity_k_dev_bi\tperplexity_k_dev_tri\tperplexity_interpolated_train\tperplexity_interpolated_dev'
        hps_to_update = [{}] if len(sys.argv) == 1 else parse_whole_updated_hps(sys.argv[1])
        for hps in hps_to_update:
            run_experiment(hps)
        print 'Time spent is {} seconds\n'.format(time.time()-start)

    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == '__main__':
    main()
