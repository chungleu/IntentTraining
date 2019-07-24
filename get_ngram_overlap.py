"""
Given a workspace, find the overlaps between ngrams between all intents in that workspace.
params:
    - n_list
    - topic

TODO: extend to multi-workspace
"""

###
import os, time
import pandas as pd
from config import *
import click
import for_csv
from logging import getLogger
logger = getLogger("get_ngram_overlap")

@click.command()
@click.argument('topic', nargs=1)
@click.argument('intent1', nargs=1)
@click.argument('intent2', nargs=1)
@click.argument('n_list', nargs=1)
@click.option('--swords', '-s', type=click.Choice(['none', 'nltk', 'config']), default='none', help="Whether to use "
"no stopwords, the nltk set, or nltk + config")

def main(topic, intent1, intent2, n_list, swords):
    def process_list_argument(list_arg):
        # TODO: move this into a separate module
        """
        An argument entered as a list will be processed as a string.
        This function transforms it into a list.
        """
        list_out = list(map(int, list_arg.strip('[]').split(',')))

        return list_out

    n_list = process_list_argument(n_list)

    from get_intent_intersections import intent_intersections

    ii = intent_intersections(n_list, stopwords_in=swords)
    
    logger.info("Importing training data..")
    ii.import_training_data(topic)
    logger.debug(ii.df_training.shape)

    logger.info("Calculating distibution of ngrams for each intent in the workspace..")
    ngram_per_intent_df, ngram_freq_df = ii.get_ngrams_per_intent()
    
    logger.info("Finding the intersections between {} and {}..".format(intent1, intent2))
    overlap_df = ii.get_intersection_freqs([intent1, intent2], ngram_freq_df)
    overlap_df = overlap_df.sort_values([intent1, intent2], ascending=False)
    
    timestr = time.strftime("%Y%m%d-%H%M")
    filename = 'ngram_overlap_' + topic + '_' + intent1 + '_' + intent2 + '_' + timestr + '_' + str(n_list).strip('[]').replace(' ', '') + '.csv'
    file_path = os.path.join(output_folder, filename)
    overlap_df.to_csv(file_path)
    logger.info('Exported csv to {}'.format(file_path))


if __name__ == '__main__':
    main()