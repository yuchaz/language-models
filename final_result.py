from package import load
from package.train import train_language_models
from package.evaluation import calc_perplexity, calc_interpolated_perplexity, calc_perplexity_without_k, update_eval_hyperparameters
from package.oov_words_handling import replace_low_freq_words
from package.hyperparameter_parser import parse_whole_updated_hps, parse_unk_section, parse_evaluation_section, parse_item_in_constant
import json

FINAL_HYPERPARAMETER_PATH = './hp_final.ini'

def main():

    hps_to_update = parse_whole_updated_hps(FINAL_HYPERPARAMETER_PATH)[0]
    training_corpus_orig = load.load_traing_corpus()
    training_corpus = replace_low_freq_words(training_corpus_orig, hps_to_update)
    language_models = train_language_models(training_corpus,hps_to_update)

    test_corpus_orig = load.load_test_corpus()
    test_corpus = replace_low_freq_words(test_corpus_orig, hps_to_update)
    perplexity = calc_interpolated_perplexity(test_corpus, language_models, hps_to_update)
    print perplexity


if __name__ == '__main__':
    main()
