from config import *
import os, time
import fuzzywuzzy.process as fuzz_process
from fuzzywuzzy import fuzz
import click
import pandas as pd
import for_csv

def get_external_data(topic):
    """
    Get and clean production data for specified topic, for out
        of scope utterance detection.
    """

    utils = for_csv.utils(topic)

    responses_todrop = utils.import_csv_to_list(responses_todrop_path)
    button_clicks = utils.import_csv_to_list(buttons_path).tolist()
    follow_up_questions = utils.import_csv_to_list(follow_up_path).tolist()
    utterances_todrop = button_clicks + other_drop + follow_up_questions

    # production data
    csv_path = os.path.join(data_dir, data_file)
    df_external = utils.import_external_data(csv_path, topic)
    process = for_csv.process_dataframe(df_external)
    process.drop_rows_with_column_value(utterance_col, utterances_todrop)
    process.remove_numeric_utterances(chars_toignore=dfclean_specialchars)
    process.remove_date_utterances()
    process.drop_duplicate_utterances(duplicate_thresh=1)
    process.drop_rows_with_column_value(response_col, responses_todrop, lower=False)

    production_df = process.get_df()

    return production_df

def get_training_data(topic):
    """
    Get and clean training data for a specified topic, for out 
    of scope utterance detection. If topic==None then a dataframe 
    with all the topics specified in config is returned, with a 
    'topic' column indicating the topic. 
    """

    if topic == None:
        df_training = join_all_training_data()
    else:
        utils = for_csv.utils(topic)

        file_name = topic + '_questions.csv'
        training_path = os.path.join(training_dir, file_name)
        df_training = utils.import_training_data(training_path)
        df_training = utils.check_questions_df_consistency(df_training, to_lower=False)

    return df_training

def join_all_training_data():
    """
    Get training data for all topics and return a dataframe with a column 'topic',
    which specifies which workspace the training comes from.
    """

    df_return = pd.DataFrame()

    for topic in workspace_list:
        tempdf = get_training_data(topic)
        tempdf['topic'] = topic

        df_return = df_return.append(tempdf)

    return df_return

class oos_failedtwice(object):
    def __init__(self, master_df, training_df):
        self.utils = for_csv.utils(topic='master')
        self.master_df = master_df
        self.training_df = training_df

    # TODO: move to external module
    def fuzzy_match_col(self, col, match_list, score_t, score_thresh=90):
        new_name, score = fuzz_process.extractOne(col, match_list, scorer=score_t)
        if score > score_thresh:
            return True
        else:
            return False

    def get_out_of_scope(self):
        print('Getting out of scope..')
        from for_csv import process_dataframe
        # import external lists
        failed_responses_list = self.utils.import_csv_to_list(master_failed_responses).tolist()

        master_df = self.master_df.copy()
        master_df['response_match'] = master_df['response'].apply(self.fuzzy_match_col, match_list=failed_responses_list, score_t=fuzz.ratio)
        self.out_of_scope_results = master_df[master_df['response_match'] == True]

        return self.out_of_scope_results

    def prioritise_out_of_scope(self):
        print('Prioritising out of scope...')

        self.out_of_scope_prioritised = self.utils.utterance_select_dissimilarity(self.out_of_scope_results, len(self.out_of_scope_results), self.training_df)

        return self.out_of_scope_prioritised

    def get_and_prioritise(self):
        self.get_out_of_scope()
        oos_prioritised = self.prioritise_out_of_scope()

        return oos_prioritised

class oos_ngrams(object):
    def __init__(self, prod_df, training_df, stopwords=[], ngram_list=[1,2]):
        self.prod_df = prod_df
        self.training_df = training_df
        self.stopwords = stopwords
        self.ngram_list = ngram_list

    def get_ngrams_from_dfs(self):
        """
        Get lists of ngrams from both dfs and store within the class 
            instance.
        """

        from for_csv.nlp import ngrams_df

        ng_prod = ngrams_df(self.prod_df, stopwords=stopwords)
        self.ngrams_prod = ng_prod.get_ngram_list(self.ngram_list)

        ng_train = ngrams_df(self.training_df, stopwords=stopwords)
        self.ngrams_train = ng_train.get_ngram_list(self.ngram_list)

        return self.ngrams_prod, self.ngrams_train

    def get_set_diff(self):
        """
        Get set diff of ngrams in training and production.
        """
        if not hasattr(self, 'ngrams_prod'):
            self.get_ngrams_from_dfs()

        return set(self.ngrams_prod) - set(self.ngrams_train)

    def get_freq_diff(self, min_freq=0):
        """
        Get frequency diff of ngrams in training and production.
        """
        
        set_diff = self.get_set_diff()

        import pandas as pd

        prod_count = pd.Series(self.ngrams_prod).value_counts()

        # filter ngrams to be only the ones that aren't in training
        prod_count = prod_count[prod_count.index.isin(set_diff)]

        prod_count = prod_count[prod_count > min_freq].sort_values(ascending=False)
        
        return prod_count

###
@click.command()
@click.argument('method', nargs=1, type=click.Choice(['failedtwice', 'ngrams']))
def get_out_of_scope(method):

    """
    get out of scope from failed twice in master, and prioritise based
    on dissimilarity from training data
    """

    def get_oos_failedtwice(display=False):
        print('Getting prod df for master..')
        master_df = get_external_data('master')

        print('Getting failedtwice utterances..')
        failedtwice = oos_failedtwice(master_df, training_df)
        out_of_scope_prioritised = failedtwice.get_and_prioritise()

        timestr = time.strftime("%Y%m%d-%H%M")
        output_filename = 'out_of_scope_failedtwice_prioritised_' + timestr + '.csv'
        out_path = os.path.join(output_folder, output_filename)
        out_of_scope_prioritised.to_csv(out_path)

        if display:
            from for_csv.topic_modelling import topic_model
            tm = topic_model(utterance_series=out_of_scope_prioritised[utterance_col], stopword_list=stopwords)
            tm.display_LDA_topics(no_topics=20, no_top_words=10)

    
    def get_oos_ngrams():
        print('Getting prod df (all)...')
        prod_df = get_external_data(topic=None)

        print('Getting set of ngrams that are out of scope..')
        oos_ng = oos_ngrams(prod_df, training_df, stopwords=stopwords, ngram_list=[3,4,5])
        freq_diff_ngrams = oos_ng.get_freq_diff(min_freq=5)  

        print(freq_diff_ngrams)

    print('Getting training df for master..')
    training_df = get_training_data('master')

    if method == 'failedtwice':
        get_oos_failedtwice()
    elif method == 'ngrams':
        get_oos_ngrams()


if __name__ == '__main__':
    get_out_of_scope()