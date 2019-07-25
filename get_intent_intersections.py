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
import for_csv
import click

@click.command()
@click.argument('topic', nargs=1)
@click.argument('n_list', nargs=1)
@click.option('--swords', '-s', type=click.Choice(['none', 'nltk', 'config']), default='none', help="Whether to use "
"no stopwords, the nltk set, or nltk + config")

def main(topic, n_list, swords):
    def process_list_argument(list_arg):
        # TODO: move this into a separate module
        """
        An argument entered as a list will be processed as a string.
        This function transforms it into a list.
        """
        list_out = list(map(int, list_arg.strip('[]()').split(',')))

        return list_out

    n_list = process_list_argument(n_list)
    
    ii = intent_intersections(n_list, stopwords_in=swords)
    
    print('Getting training data for ' + topic + '...')
    ii.import_training_data(topic)

    print('Calculating ngram intersections...')
    intersection_df, intersection_size_df = ii.calculate_ngram_intersections()
    
    timestr = time.strftime("%Y%m%d-%H%M")
    filename = 'ngram_intersection_matrix_' + topic + '_' + timestr + '_' + str(n_list).strip('[]').replace(' ', '') + '.csv'
    file_path = os.path.join(output_folder, filename)
    intersection_size_df.to_csv(file_path)
    print('Exported csv to ' + file_path)


class intent_intersections(object):
    def __init__(self, n_list, stopwords_in=[], intent_col='Intent'):
        self.n_list = n_list
        self.stopwords = self.process_stopwords_arg(stopwords_in)
        self.intent_col = intent_col

    def process_stopwords_arg(self, stopwords_arg):
        """
        'default' -> nltk + config
        'none' -> no stopwords used at all
        'nltk' -> just nltk
        """

        if stopwords_arg == 'none':
            return '_none'
        elif stopwords_arg == 'nltk':
            return None
        elif stopwords_arg == 'config':
            return stopwords

    def import_training_data(self, topic):
        """
        gets and cleans training data from file specified in config.
        also creates a list of the intents within training
        TODO: pull this out into separate import module
        """
        file_name = topic + '_questions.csv'

        training_path = os.path.join(training_dir, file_name)
        ut = for_csv.utils(topic)
        df_training = ut.import_training_data(training_path)
        df_training = ut.check_questions_df_consistency(df_training, to_lower=False)
        
        self.intents = df_training[self.intent_col].unique()
        self.df_training = df_training

    def import_training_df(self, train_df):
        """
        Instead of using import_training data, you can also just use an existing dataframe directly
        in the class instance.
        Topic set to none
        """
        
        ut = for_csv.utils(None)
        train_df = ut.check_questions_df_consistency(train_df, to_lower=False, intent_col=self.intent_col)

        self.intents = train_df[self.intent_col].unique()
        self.df_training = train_df

    def get_ngrams_per_intent(self):
        """
        Creates:
            - ngram_per_intent_df: dataframe with intents as columns, list of all ngrams in each column
            - ngram_freq_df: dataframe with frequencies per ngram (row) and intent (column)
        """

        df_training = self.df_training
        
        ngram_dict = dict()
        ngram_freq_list = []

        for intent in self.intents:
            df_intent = df_training[df_training[self.intent_col] == intent]

            ngrams = for_csv.nlp.ngrams_df(df_intent, stopwords_in=self.stopwords, utterance_col=utterance_col, chars_remove=chars_remove)

            ngram_dict[intent] = pd.Series(ngrams.get_ngram_list(self.n_list), name=intent)
            ngram_freq_list.append(ngram_dict[intent].value_counts().to_frame(name=intent))

        self.ngram_per_intent_df = pd.DataFrame.from_dict(ngram_dict).dropna()
        self.ngram_freq_df = pd.concat([frame for frame in ngram_freq_list], sort=False).sum(level=0)

        return self.ngram_per_intent_df, self.ngram_freq_df

    def calculate_ngram_intersections(self):
        """
        Creates two dataframes, each symmetric dataframes with intents as both rows and columns:
            - intersection_df: contains set of all ngrams in each intersection
            - intersection_size_df: contains size of (no. ngrams in) each intersection
        """
        try:
            self.ngram_per_intent_df
        except:
            self.get_ngrams_per_intent()

        intents = self.intents
        intent_pairs = [(x,y) for x in intents for y in intents if x != y]

        intersection_df = pd.DataFrame(index=intents, columns=intents)
        intersection_size_df = pd.DataFrame(index=intents, columns=intents)

        for pair in intent_pairs:
            set1 = set(self.ngram_per_intent_df[pair[0]])
            set2 = set(self.ngram_per_intent_df[pair[1]])
            intersection_df.loc[pair[0], pair[1]] = set1.intersection(set2)
            intersection_size_df.loc[pair[0], pair[1]] = len(intersection_df.loc[pair[0], pair[1]])

        self.intersection_df = intersection_df
        self.intersection_size_df = intersection_size_df

        return self.intersection_df, self.intersection_size_df

    def get_intersection_freqs(self, intent_pair, ngram_freq_df):
        """
        Return a dataframe containing the specified ngrams and their frequencies in particular intents.
        """
        try:
            self.intersection_df
        except:
            self.calculate_ngram_intersections()

        ngrams = self.intersection_df.loc[intent_pair[0], intent_pair[1]]

        return ngram_freq_df.loc[ngrams, intent_pair]


####
if __name__ == '__main__':
    main()