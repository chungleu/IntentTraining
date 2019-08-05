"""
Accepts blind set output, intents which are confused, n_list, whether to retain intent order.
Splits utterances into ngrams, searches for examples in that workspace’s training for these ngrams (prioritising the intents chosen)
"""
import sys
sys.path.append('..')

import click
import pandas as pd
import time
import os
import for_csv
from for_csv.utils import process_list_argument
from for_csv.nlp import extract_ngrams
from cli_tools.get_utterances_containing import get_utterances_containing
from config import *
from logging import getLogger
logger = getLogger("diagnose_confusion")

@click.command()
@click.argument("utterance", nargs=1)
@click.argument("topic", nargs=1)
@click.argument("n_list", nargs=1)
@click.option('--swords', '-s', type=click.Choice(['none', 'nltk', 'config']), default='none', help="Whether to use "
"no stopwords, the nltk set, or nltk + config")
def click_main(utterance, topic, n_list, swords):
    logger.info("Finding matching utterances..")
    logger.debug(utterance)
    n_list = process_list_argument(n_list, int)
    results_df = diagnose_confusion(utterance, topic, n_list, swords)

    logger.info("{} utterances found in {} intents".format(len(results_df), len(results_df['Intent'].unique())))

    if len(results_df) > 0:
        timestr = time.strftime("%Y%m%d-%H%M")
        filename = 'diagnose_confusion_' + topic + '_' + timestr + '_' + str(n_list).strip('[]').replace(' ', '') + '.csv'
        file_path = os.path.join(output_folder, filename)
        results_df.to_csv(file_path)
        logger.info("Results exported to {}".format(file_path))
    else:
        logger.warn("No results found so nothing exported.")

def diagnose_confusion(utterance, topic, n_list, stopwords_in):
    """
    Takes an utterance, splits it into ngrams, and searches for these ngrams in training. Returns training samples containing these ngrams.
    Intended as a rough, but more explainable version of looking at the nearest training utterances in feature space.
    Args:
    - utterance: to be split and searched.
    - topic: where to look for training.
    - n_list: comma separated list of which length ngrams to split into.
    - stopwords_in: one of 'none', 'nltk', or 'config'.
    """
    # TODO: could fuzzy match ngrams for better approximation
    ##

    # process args
    stopwords_list = process_stopwords_arg(stopwords_in)

    # make list of all ngrams in utterance
    ngrams_in_utterance = []

    for n in n_list:
        ngrams_in_utterance.extend( extract_ngrams(utterance, n, stopwords_list=stopwords_in, chars_remove=chars_remove) )

    # look for each of these ngrams in training
    results_df = pd.DataFrame()
    for ngram in ngrams_in_utterance:
        tempdf = get_utterances_containing(ngram, topic, case_sensitive=False, just_utterances=False, to_csv=False, training=True, hide_output=True).drop(columns='utterance_lower')
        results_df = results_df.append(tempdf)

    results_df = results_df.drop_duplicates(subset='utterance')

    # add a col which says which ngrams caused trouble
    results_df['ngrams found'] = ""
    for idx, row in results_df.iterrows():
        temp_ngrams_in_training_utterance = [ngram for ngram in ngrams_in_utterance if ngram in row['utterance'].lower()]
        results_df.loc[idx, 'ngrams found'] = str(temp_ngrams_in_training_utterance).strip("[]")

    return results_df.sort_values('Intent')

def process_stopwords_arg(stopwords_arg):
        """
        'default' -> nltk + config
        'none' -> no stopwords used at all
        'nltk' -> just nltk
        """

        # TODO: this shouldn't exist - it's just a (weird) mapping of variables. also exists in intent_intersections

        if stopwords_arg == 'none':
            return '_none'
        elif stopwords_arg == 'nltk':
            return None
        elif stopwords_arg == 'config':
            return stopwords


if __name__ == "__main__":
    click_main()
    