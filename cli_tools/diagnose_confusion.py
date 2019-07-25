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
from get_utterances_containing import get_utterances_containing
from config import *
from logging import getLogger
logger = getLogger("diagnose_confusion")

@click.command()
@click.argument("blind_result_path", nargs=1)
@click.argument("topic", nargs=1)
@click.argument("n_list", nargs=1)
@click.option("--expected_intent", "-ei", type=str, help="Expected intent")
@click.option("--intent1", "-i1", type=str, help="Intent1")
@click.option("-retain_intent_order", "-o", is_flag=True)

def main(blind_result_path, topic, n_list, expected_intent, intent1, retain_intent_order):
    n_list = process_list_argument(n_list, int)
    stopwords_in = '_none'

    logger.info("Opening results file..")
    results_df = pd.read_csv(blind_result_path)

    if any(["Question", "Expected Intent", "Intent1"]) not in results_df.columns.values:
        raise ValueError('At least one of the columns Question, Expected Intent, Intent1 is not in ')
    
    logger.info("Getting utterances from conflict..")
    # TODO: get retain_intent_order argument working
    intent_list = [expected_intent, intent1]
    intent_list_lower = [item.lower() for item in intent_list]
    conflict_df = results_df[results_df['Expected Intent'].str.lower().isin(intent_list_lower) & results_df['Intent1'].str.lower().isin(intent_list_lower) & (results_df['Expected Intent'] != results_df['Intent1'])]
    logger.info("{} utterances found from conflict".format(len(conflict_df)))

    logger.info("Extracting ngrams..")
    ng = for_csv.nlp.ngrams_df(conflict_df, stopwords_in=stopwords_in, chars_remove=chars_remove, utterance_col='Question')
    ngram_list = ng.get_ngram_list(n_list)
    # TODO: (started below) format into:
    # utterance | expected intent | intent1 | potential problem training samples (EI) | potential problem training samples (I1) | ngrams not in training
    df_with_ngrams = ng.create_ngram_cols(n_list) # this has the columns Question | Expected Intent | Intent1 | ... | [ngram_i]n

    logger.info("Finding occurrences of ngrams in training..")
    return_df = pd.DataFrame()

    # iterate through each conflicting example, and get all examples where the ngrams appear
    for row in df_with_ngrams.iterrows():
        row_ngram_list = row[1]['ngrams_all']
        row_df = pd.DataFrame()

        ngrams_not_in_training = []

        # get training samples containing ngrams, and append to a big df
        for ngram in row_ngram_list:
            tempdf = get_utterances_containing(ngram, topic, case_sensitive=False, just_utterances=False, to_csv=False, training=True, hide_output=True).drop(columns='utterance_lower')
            tempdf = tempdf[tempdf['Intent'].isin(intent_list_lower)]

            if len(tempdf) == 0:
                ngrams_not_in_training.append(ngram)

            row_df = row_df.append(tempdf)

        row_df = row_df.drop_duplicates()

        cols_transfer = ["Question", "Expected Intent", "Intent1", "Confidence1", "Confusion"]

        for col in cols_transfer:
            try:
                row_df[col] = str(row[1][col])
            except:
                logger.warn("Column {} not in input file so won't appear in output.".format(col))

        # add any ngrams not in any of the training to a separate column
        row_df['ngrams not in training'] = str(ngrams_not_in_training)

        # make list of ngrams that are in both original and training utterances
        problem_ngrams_all = []
        for nrow in row_df.iterrows():
            problem_ngrams = [ngram for ngram in row_ngram_list if (ngram in nrow[1]['Question'].lower() and ngram in nrow[1]['utterance'].lower())]
            problem_ngrams_all.append(problem_ngrams)

        row_df['problem_ngrams'] = problem_ngrams_all

        return_df = return_df.append(row_df, sort=False)

    return_df = return_df[["Question", "Expected Intent", "Intent1", 'utterance', 'Intent', 'problem_ngrams', 'ngrams not in training']].rename(columns={'utterance': 'training utterance', 'Intent': 'training intent', 'Question': 'confused utterance'})
    return_df = return_df.sort_values(['confused utterance', 'training intent'])

    timestr = time.strftime("%Y%m%d-%H%M")
    filename = 'diagnose_confusion_' + topic + '_' + expected_intent + '_' + intent1 + '_' + timestr + '.csv'
    file_path = os.path.join(output_folder, filename)
    return_df.to_csv(file_path, index=False)
    logger.info("Exported results to CSV at {}".format(file_path))

if __name__ == "__main__":
    main()