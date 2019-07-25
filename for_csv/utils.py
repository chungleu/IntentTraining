# local dependencies
import os
from .minhash_string_similarity import MinHash_similarity

# external dependencies
import pandas as pd
import numpy as np
import re
import time

# TODO: implement management of column names from config

def process_list_argument(list_arg, val_type=int):
    """
    An argument entered as a list will be processed as a string.
    This function transforms it into a list.
    """
    list_out = list(map(val_type, list_arg.strip('[]').split(',')))

    return list_out

class utils(object):
    def __init__(self, topic, margin_params={'margin_max': 0.5, 'min_conf1':0.2}, lowconf_max=0.1, dissimilarity_params = {'dissimilarity_min': 0, 'sortby_margin': True}, minhash_params={'threshold':0.5, 'num_perm':512, 'shingle_length':5}):
        self.margin_params = margin_params
        self.lowconf_max = lowconf_max
        self.dissimilarity_min = dissimilarity_params['dissimilarity_min']
        self.dissimilarity_sortby_margin = dissimilarity_params['sortby_margin']
        self.topic = topic
        self.minhash_params = minhash_params

    def import_external_data(self, data_path, topic=None):
        """
        Pulls csv extract into dataframe
        """
        df = pd.read_csv(data_path, low_memory=False, encoding = "ISO-8859-1")

        if topic:
            print('Filtered external data to just show topic ' + topic)
            df = df[df['modelRef_0'] == topic]

        return df

    def import_training_data(self, data_path):
        """
        get data from csv with utterance, intent columns
        """
        questions = pd.read_csv(
            data_path, names=['utterance', 'Intent', 'TestSetRef'])
        #print('Imported ' + data_path + '(' + str(len(questions)) + ')')

        questions = self.check_questions_df_consistency(questions)

        return questions

    def import_csv_to_list(self, data_path):
        """
        import one column csv to pandas series
        """
        return pd.read_csv(data_path, squeeze=True)

    def check_questions_df_consistency(self, questions, to_lower=False, utterance_col='utterance', intent_col='Intent'):
        # TODO: to_lower results in posting 0 utterances to training. Why?
        if to_lower:
            questions = questions.apply(lambda x: x.astype(str).str.lower())

        questions[intent_col] = questions[intent_col].astype(str).str.lower()
        length_before = len(questions)

        if questions.apply(lambda x: x.astype(str).str.lower()).duplicated(subset=[utterance_col, intent_col]).any() == True:
            #duplicates = questions[questions.duplicated(subset=['utterance','Intent'])]
            #print('List of duplicate utterances to be disregarded:')
            # print(set(list(duplicates.Question)))
            questions_lower = questions.apply(
                lambda x: x.astype(str).str.lower())
            dup_inds = questions_lower[questions_lower.duplicated(
                subset=utterance_col, keep='first')].index
            questions = questions.drop(index=dup_inds)
            #print('Dataset now formed of distinct Utterance-Intent pairs.')

        if questions[pd.isnull(questions[intent_col])].shape[0] > 0:
            #print('List of utterances with empty intents to be disregarded:')
            #questions_list = list(questions[pd.isnull(questions['Intent'])]['utterance'])
            # print(questions_list)
            questions['TestSetRef'] = 'temporary value'
            questions = questions.dropna()
            questions['TestSetRef'] = np.nan
            #print('Dataset now formed by fully-populated Utterance-Intent pairs.')

        if questions[pd.isnull(questions[utterance_col])].shape[0] > 0:
            #print('Empty utterances found:')
            # print(questions[pd.isnull(questions['utterance'])])
            questions = questions.dropna(subset=[utterance_col])

        elements = list(questions[utterance_col])
        e_list = []

        for e in elements:
            try:
                if (re.sub('[ -~]', '', e)) != "":
                    e_list.append(e)
            except:
                print('Error with non-ASCII detection: ' + str(e))

        #print("Non-ASCII characters are detected within " + str(len(e_list)) + " utterances.")

        if 'TestSetRef' in questions.columns.values:
            questions = questions.drop(columns='TestSetRef')

        length_after = len(questions)
        if length_after < length_before:
            print('Whilst checking df consistency ' + str(length_before-length_after) + ' utterances were dropped.')

        return questions

    def train_test_split(self, questions, sampleSize, stratified=True, return_split=True):
        """
        creates a binary indicator column which randomly splits a dataframe into training and test sets.
        sampleSize dictates size of the training set (TestSetRef_1 == 0)
        0 = training (to push to workspace), 1 = test (to keep)
        """
        intents = questions['Intent'].unique().tolist()
        
        # add indicator column where 0 means push to workspace, 1 means remaining data
        indicator_col = "TestSetRef_1"
        questions[indicator_col] = 1

        if stratified == True:
            # stratified sampling
            samples_per_intent = np.ceil(
                questions.groupby('Intent').count() * sampleSize)

            for intent in intents:
                questions_intent = questions[questions['Intent'] == intent]
                no_tosample = int(samples_per_intent['utterance'][intent])

                # random sampling
                sample_inds = questions_intent.sample(n=no_tosample).index
                questions.loc[sample_inds, indicator_col] = 0

        elif stratified == False:
            # random sampling
            sample_inds = questions.sample(frac=sampleSize).index
            questions.loc[sample_inds, indicator_col] = 0

        if return_split:
            questions_train = questions[questions['TestSetRef_1'] == 0].copy()
            questions_test = questions[questions['TestSetRef_1'] == 1].copy()

            return questions_train, questions_test
        else:
            return questions

    def get_priority_utterances(self, no_utterances, external_data, questions_in_training, method):
        """
        Switching function which calls other functions to get priority utterances
        """

        if method == 'random':
            questions = external_data.sample(no_utterances)

        elif method == 'lowconf':
            questions = self.utterance_select_lowconf(
                external_data, no_utterances)

        elif method == 'margin':
            questions = self.utterance_select_margin(
                external_data, no_utterances)

        elif method == 'similarity':
            questions = self.utterance_select_dissimilarity(
                external_data, no_utterances, questions_in_training)

        #elif method == 'small_intents':
            #questions = self.utterance_select_smaller_intents(
            #utterance_list, no_utterances, questions_in_training)
            
        else:
            print('Method not defined.')

        return questions

    def calculate_margin(self, df, conf1_col="confidence1_0", conf2_col="confidence2_0"):
        """
        Creates a column named 'margin' which is the difference between conf@1
        and conf@2.
        """

        results = df.copy()
        results['margin'] = results[conf1_col] - results[conf2_col]

        results = results[results['margin'] > 0] # fix for 0 margins which shouldn't be in the csv extract

        return results

    def utterance_select_margin(self, external_data, no_utterances):
        """
        Based on these results, select the utterances which have the lowest margin (0 < M <= T)
        Optional min_conf to specify minimum confidence level utterances to be selected.
        Return these utterances and their intents in a dataframe
        """

        min_conf1 = self.margin_params['min_conf1']
        margin_max = self.margin_params['margin_max']

        # TODO: put an upper limit on the margin as a condition for returning utterances?
        results = self.calculate_margin(external_data)

        results = results[results['confidence1_0'] >= min_conf1]

        questions = results.sort_values('margin', ascending=True).head(no_utterances)

        # filter using self.lowconf_max
        size_before_filter = len(questions)
        questions = questions[questions['margin'] <= margin_max]
        size_after_filter = len(questions)

        if size_after_filter < size_before_filter:
            print('Only ' + str(size_after_filter) + '/' + str(no_utterances) + ' utterances present with a margin below the max of ' + str(self.lowconf_max))

        print(str(no_utterances) + ' utterances returned with an average margin of ' +
              str(questions['margin'].mean()))

        return questions

    def utterance_select_lowconf(self, external_data, no_utterances, ignore_zero=True, conf1_col = 'confidence1_0'):
        """
        Select the utterances which have the lowest confidence@1.
        By default ignores utterances with confidence@1==0 (ignore_zero)
        """
        # TODO: maybe put an upper limit on the confidence that can be returned here
        results = external_data.copy()

        if ignore_zero:
            results = results[results[conf1_col] > 0]
        questions = results.sort_values(
            conf1_col, ascending=True).head(no_utterances)

        # filter using self.lowconf_max
        size_before_filter = len(questions)
        questions = questions[questions[conf1_col] <= self.lowconf_max]
        size_after_filter = len(questions)

        if size_after_filter < size_before_filter:
            print('Only ' + str(size_after_filter) + '/' + str(no_utterances) + ' utterances present with a confidence@1 below the max of ' + str(self.lowconf_max))
       
        print(str(no_utterances) + ' utterances returned with an average confidence@1 of ' +
            str(questions[conf1_col].mean()))

        return questions

    def utterance_select_dissimilarity(self, external_data, no_utterances, questions_in_training):
        """
        Select no_utterances from utterance_list prioritised by their dissimilarity from the training corpus
        (questions_in_training).
        """
        # get number of utterances in the training set that each available utterances is similar to
        # TODO: is there a better measure of dissimilarity?

        minhash_params = self.minhash_params
        mh = MinHash_similarity(threshold=minhash_params['threshold'], num_perm=minhash_params['num_perm'],
            shingle_length=minhash_params['shingle_length'])
        df_query = mh.similarity_threshold_bulk(
            df_library=questions_in_training, df_query=external_data, return_df=True)

        df_query = df_query[df_query['no_similar'] >= self.dissimilarity_min]
        no_available = len(df_query)

        if no_utterances > no_available:
            print('Not enough utterances available. Returned all ' + str(no_available) + ' available utterances.')
            no_utterances = no_available
        
        no_similar_to_zero = len(df_query[df_query['no_similar']==0])

        questions = df_query.sort_values('no_similar', ascending=True).head(no_utterances)

        if self.dissimilarity_sortby_margin:
            questions = self.sort_df_by_margin(questions)

        print(str(no_utterances) + ' utterances returned which are similar to between ' +
            str(questions['no_similar'].min()) + ' and ' +  str(questions['no_similar'].max()) + 
            ' training utterances.')

        if no_similar_to_zero > no_available*0.25:
            print('Warning: more than 25% (' + str(no_similar_to_zero) + '/' + str(no_available) + ') of utterances in the retrieved dataset '
            'are similar to no more than 0 utterances in training. Maybe consider lowering the minhash similarity threshold.')

        return questions

    def utterance_select_smaller_intents(self, utterance_list, no_utterances, questions_in_training):
        """
        Prioritise utterances based on number in training - load utterances into smaller intents first.
        """
        tempdf = questions_in_training.copy()

        # calculate the probability of selection of each intent
        intent_amounts = tempdf['Intent'].value_counts()
        relative_probs = 1 / intent_amounts
        norm_probs = relative_probs / relative_probs.sum()

        # fill dataframe col in utterance_list with these probabilities
        utterance_list['norm_prob'] = 0
        for idx, row in utterance_list.iterrows():
            try:
                utterance_list.loc[idx, 'norm_prob'] = norm_probs[row['Intent']]
            except:
                # if can't find the column, then just fill with mean
                utterance_list.loc[idx, 'norm_prob'] = norm_probs.mean()

        # take a weighted sample based on these probabilities
        return utterance_list.sample(n=no_utterances, weights='norm_prob')[['utterance', 'Intent']]

    def sort_df_by_margin(self, df, ascending=True):
        """
        Sorts df by column chosen. If column doesn't exist, then create it first.
        Only works for margin at the moment
        """

        if 'margin' not in df.columns:
            df = self.calculate_margin(df)
        
        df = df.sort_values('margin', ascending=True)

        return df

    def df_select_specific_intents(self, external_df, intents_to_select, include_second_intent=True):
        """
        Filter prioritised dataframe by intent. Not case sensitive.
        - intents_to_select: list of intents for which records should be retrieved.
        - include_second_intent: takes into account the intent with the second highest confidence when 
        checking for a match.
        """

        intents_to_select = [intent.lower() for intent in intents_to_select]

        if include_second_intent:
            return external_df[external_df['intent1_0'].str.lower().isin(intents_to_select) | external_df['intent2_0'].str.lower().isin(intents_to_select)]
        else:
            return external_df[external_df['intent1_0'].str.lower().isin(intents_to_select)]

