"""
Get top ngrams for all intents in a topic. Outputs as a CSV with columns n | ngram | frequency | intent
Takes topic, n_list as arguments.
n_list should be specified as a comma-separated list of values, WITHOUT SPACES
"""

## ARGS FOR CLICK
top_a = False
#norm = True
norm_thresh = False
###
import sys
sys.path.append('..')

import os, time
import click
import pandas as pd
from config import *
import for_csv

@click.command()
@click.argument('topic', nargs=1)
@click.argument('n_list', nargs=1)
@click.option('--absolute', '-a', is_flag=True, help='Return absolute values (instead of normalised)')

def main(topic, n_list, absolute):

    def process_list_argument(list_arg):
        """
        An argument entered as a list will be processed as a string.
        This function transforms it into a list.
        """
        list_out = list(map(int, list_arg.strip('[]').split(',')))

        return list_out

    if not absolute:
        norm = True
    else:
        norm = False

    n_list = process_list_argument(n_list)
    utils = for_csv.utils(topic)

    # import training data for topic, and clean
    print('Importing training data for topic ' + topic + '...')
    file_name = topic + '_questions.csv'

    training_path = os.path.join(training_dir, file_name)
    df_training = utils.import_training_data(training_path)
    df_training = utils.check_questions_df_consistency(df_training, to_lower=False)

    # get ngrams for each intent and append to main dataframe
    print('Getting ngrams for each intent..')
    intents = df_training['Intent'].unique()
    ngram_freq_dict = pd.DataFrame()

    for intent in intents:
        df_intent = df_training[df_training['Intent'] == intent]
        # TODO: is there an overhead in creating a new class instance for each intent?
        ngrams = for_csv.nlp.ngrams_df(df_intent, stopwords=stopwords, utterance_col=utterance_col, chars_remove=chars_remove)
        temp_freq_dict = ngrams.get_ngram_frequencies(n_list, top_a, norm, norm_thresh)
        temp_freq_dict['intent'] = intent

        ngram_freq_dict = ngram_freq_dict.append(temp_freq_dict)

    timestr = time.strftime("%Y%m%d-%H%M")
    filename = 'ngrams_' + topic + '_' + timestr + '_' + str(n_list).strip('[]').replace(' ', '') + '.csv'
    file_path = os.path.join(output_folder, filename)
    ngram_freq_dict.to_csv(file_path, index=False)
    print('Exported csv to ' + file_path)

if __name__ == '__main__':
    main()