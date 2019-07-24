"""
Takes a topic as input, and checks the consistency of the test set against training.
Features:
    - % utterances per intent (test set vs training)
    - flag utterances in test set that are similar to training
    - check consistency between intent names in test set and intent names in training

Args: train_path, test_path

"""
# TODO: check consistency against what's in production.
# TODO: this seems like it would work better as an sklearn pipeline?

# external
import pandas as pd
import numpy as np
import click
import os, sys
from tqdm import tqdm
tqdm.pandas()

# internal
from config import *
import for_csv
from for_csv.nlp import fuzzy_match_lists
from logging import getLogger
logger = getLogger('check_blindset')

@click.command()
@click.option('--train_path', '-tr', type=str)
@click.option('--test_path', '-te', type=str)

def main(train_path, test_path):
    debug = True

    logger.info('Importing data')
    df_train = import_data(train_path)
    df_test = import_data(test_path)

    cb = check_blindset(df_train, df_test, debug=debug)

    cb.display_set_sizes(cb.df_train, cb.df_test)
    cb.check_test_intent_names() # NEEDS TO BE FIRST
    cb.check_test_for_duplicates()
    cb.check_for_duplicates_between_test_train()
    cb.check_fuzzy_duplicates_between_test_train()
    cb.check_test_set_size_overall()
    cb.check_test_set_size_intents()
    cb.display_set_sizes(cb.df_train, cb.df_test_final)
    cb.export_test_df_with_recommendations()

def import_data(path):
    """
    Given a path of training/test data, will return a dataframe with 
    appropriately named columns.
    """

    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, path)

    df = pd.read_csv(
        abs_file_path, names=['utterance', 'Intent'])

    return df


class check_blindset(object):
    def __init__(self, df_train, df_test, test_set_minsize=0.25, size_intent_allowance = 0.2, debug=False):
        self.df_train = df_train
        self.df_test = df_test
        self.test_set_minsize = test_set_minsize
        self.minsize_per_intent = test_set_minsize*(1-size_intent_allowance)
        self.intents_in_training = set(self.df_train['Intent'])
        self.intents_in_test = set(self.df_test['Intent'].unique())
        self.check_failed = False
        self.debug = debug
        self.df_test_final = self.df_test.copy() # dataframe to modify and return

        # some flags for working out whether functions have run (use a pipeline!)
        self.run_check_test_for_duplicates = False

    def display_set_sizes(self, train, test):
        """
        Returns the initial training and test set sizes to the user.
        Intended to be used before any processing is done.
        """
        test_set_size = len(test)
        train_set_size = len(train)

        test_train_ratio = test_set_size / train_set_size

        logger.info("Training size: {}, Test set size: {}, Ratio: {}".format(train_set_size, test_set_size, test_train_ratio))

    def check_test_intent_names(self):
        """
        Checks whether all intent names in test set exist in training.
        If any don't, directs the user to a CSV indicating which intent
        names are erroneous.
        """
        logger.info('Checking all intents in test set exist in training..')

        # TODO: could autocorrect wrong-cased intent names automatically.
        intents_in_test_not_in_train = self.intents_in_test - self.intents_in_training
        no_intents_missing = len(intents_in_test_not_in_train)

        if no_intents_missing > 0:
            self.df_test_final['incorrect_intent_label'] = self.df_test_final['Intent'].isin(intents_in_test_not_in_train)

            logger.warn('The following ' + str(len(intents_in_test_not_in_train)) + ' intent names exist in the test set and not in training: ' + str(intents_in_test_not_in_train) + ' '
            '(incorrect_intent_label)')

            self.check_failed = True

    def check_training_intents_in_test(self):
        """
        Checks whether all intents in training are in test.
        """
        logger.warn('check_training_intents_in_test is legacy: use check_test_set_size_intents to check for both small and empty intents in the test set.')

        intents_in_train_not_in_test = self.intents_in_training - self.intents_in_test
        no_intents_missing = len(intents_in_train_not_in_test)

        if no_intents_missing > 0:
            logger.warn('The following ' + str(no_intents_missing) + ' intents exist in training but have no examples '
            'in the test set: ' + str(intents_in_train_not_in_test))
            self.check_failed = True

    def check_test_set_size_overall(self):
        """
        Checks the overall size of the test set is greater than the limit specified in
        self.test_set_minsize.
        """

        if (self.check_failed == True) & (self.debug == False): 
            logger.error("Test set size checks will only run when there are no clashes in intent names.")
            sys.exit()

        test_set_size_ratio = len(self.df_test) / len(self.df_train)

        if test_set_size_ratio < self.test_set_minsize:
            logger.warn('Overall test set is ' + str(round(test_set_size_ratio*100, 1)) + '% of the size of training. '
            'The recommended size is ' + str(self.test_set_minsize*100) + '% of training. It is strongly recommended to fetch more test examples.')

    def check_test_set_size_intents(self):
        """
        Checks that the test set for each intent is greater than test_set_minsize for
        each intent.
        Assumes that the labels in the test set are the same as in the training set.
        """

        logger.info('Checking the size of each intent in the test set..')

        if (self.check_failed == True) & (self.debug == False): 
            logger.error("Test set size checks will only run when there are no clashes in intent names.")
            sys.exit()

        train_intentdist = self.df_get_intent_distribution(self.df_train, norm=False)
        test_intentdist = self.df_get_intent_distribution(self.df_test, norm=False)

        test_train_ratio = test_intentdist / train_intentdist

        intents_in_train_not_in_test = self.intents_in_training - self.intents_in_test
        test_train_ratio[intents_in_train_not_in_test] = 0

        too_small_intents = test_train_ratio[test_train_ratio < self.minsize_per_intent]
        no_too_small_intents = len(too_small_intents)

        if no_too_small_intents > 0:
            error_df = pd.DataFrame(index=too_small_intents.index)
            error_df['in test'] = test_intentdist[test_intentdist.index.isin(too_small_intents.index)]
            error_df['in test'] = error_df['in test'].fillna(0)
            error_df['total amount'] = np.ceil(train_intentdist[train_intentdist.index.isin(too_small_intents.index)] * self.minsize_per_intent)
            error_df['number to get'] = error_df['total amount'] - error_df['in test']

            output_path = os.path.join(output_folder, 'test_set_intents_too_small.csv')
            error_df.to_csv(output_path)

            logger.warn('There are ' + str(no_too_small_intents) + ' intents with insufficient test data. '
            'See more details in the CSV at ' + output_path)

    def check_test_for_duplicates(self):
        """
        Automatically removes any [utterance, Intent] duplicates and signals this to the user.
        Flags any duplicates of utterance only and outputs to CSV.
        """
        logger.info('Checking for exact duplicates in test set..')

        # utterance & Intent duplicates
        df_temp_drop_duplicates = self.df_test_final.drop_duplicates(subset=['utterance', 'Intent'])
        no_dropped = len(self.df_test_final) - len(df_temp_drop_duplicates)

        if no_dropped > 0:
            logger.warn(str(no_dropped) + ' duplicate rows dropped from test set')
            self.df_test_final = df_temp_drop_duplicates

        # utterance only duplicates
        df_utterance_duplicates = self.df_test_final[self.df_test_final.duplicated(subset='utterance', keep=False)]
        no_duplicate_utterances = len(df_utterance_duplicates)
        
        if no_duplicate_utterances > 0:
            logger.warn(str(no_duplicate_utterances) + ' duplicate utterances which belong to different intents found in test set. '
            '(duplicate_utterance)')
            self.df_test_final['duplicate_utterance'] = self.df_test_final.duplicated(subset='utterance', keep=False)

        self.run_check_test_for_duplicates = True

    def check_for_duplicates_between_test_train(self):
        """
        Automatically removes [utterance, Intent] pairs in test set that exist in training.
        Flags if any utterances exist in test but are tagged to a different intent in training.
        """
        if not self.run_check_test_for_duplicates:
            self.check_test_for_duplicates()

        logger.info('Checking for exact duplicates between training and test..')
        
        # utterance & Intent duplicates
        df_test_temp = self.df_test_final.copy()
        df_train_temp = self.df_train.copy()
        df_test_temp['set'] = 'test'
        df_train_temp['set'] = 'train'
        df_all_temp = df_test_temp.append(df_train_temp, sort=False)
        df_all_temp['utterance'] = df_all_temp['utterance'].str.lower()
        duplicated = df_all_temp.duplicated(subset=['utterance', 'Intent'], keep=False)

        if len(duplicated) > 0:
            dup_in_df = df_all_temp[duplicated]
            test_idx_to_drop = dup_in_df[dup_in_df['set'] == 'test'].index
            self.df_test_final = self.df_test_final.drop(test_idx_to_drop)
            logger.warn(str(len(test_idx_to_drop)) + ' records have been dropped from test that are also present in training.')
        
        # TODO: utterance duplicates (different intents) -> new function

    def check_fuzzy_duplicates_between_test_train(self):
        """
        Goes through each utterances in the test set and checks if there is a highly similar utterance
        in training.
        """
        logger.info('Checking for approximate duplicates between train and test..')
        
        df_test_similaritytest = self.df_test_final.copy()
        train_utterances = self.df_train['utterance']

        df_test_similaritytest['similar_to_training'] = df_test_similaritytest['utterance'].progress_apply(fuzzy_match_lists, match_list=train_utterances, return_name=True)
        
        no_similar = len(df_test_similaritytest[df_test_similaritytest['similar_to_training'] != False])

        if no_similar > 0:
            self.df_test_final['similar_to_training'] = df_test_similaritytest['similar_to_training']
            logger.warn(str(no_similar) + '/' + str(len(df_test_similaritytest)) + ' utterances in the test set are very similar to an utterance in training. '
            '(similar_to_training)')

    def df_get_intent_distribution(self, df, norm=True):
        """
        Given a dataframe, returns a series of the distribution of utterances over intent. 
        """

        groupby = df.groupby('Intent').count()['utterance']

        if norm:
            groupby = groupby/groupby.sum()

        return groupby

    def export_test_df_with_recommendations(self):
        """
        Exports final df to csv.
        """
        output_path = os.path.join(output_folder, 'test_set_recommendations.csv')
        self.df_test_final.to_csv(output_path)
        logger.info('Test set annotated with recommendations has been exported to ' + output_path)

###
if __name__ == '__main__':
    main()