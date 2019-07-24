# local dependencies
import os
from __init__ import DATA_FOLDER, OUTPUT_FOLDER
from ConversationTestKitMin import MonteCarloV1, BlindTestV1
from Credentials import ctx, conversation_version, active_adoption
from minhash_string_similarity import MinHash_similarity
import summarise_results

# external dependencies
import pandas as pd
import numpy as np
import re
import time

mc = MonteCarloV1(url=ctx.get(active_adoption)['url'], username=ctx.get(active_adoption)[
                  'username'], password=ctx.get(active_adoption)['password'], version=conversation_version)
bt = BlindTestV1(url=ctx.get(active_adoption)['url'], username=ctx.get(active_adoption)[
                 'username'], password=ctx.get(active_adoption)['password'], version=conversation_version)


class run_test(object):
    def __init__(self, margin_max, lowconf_max, topic):
        self.margin_max = margin_max
        self.lowconf_max = lowconf_max
        self.topic = topic

    def import_data(self, data_path):
        """
        get data from csv with utterance, intent columns
        """
        questions = pd.read_csv(
            data_path, names=['Question', 'Intent', 'TestSetRef'])
        #print('Imported ' + data_path + '(' + str(len(questions)) + ')')

        questions = self.check_questions_df_consistency(questions)

        return questions

    def check_questions_df_consistency(self, questions, to_lower=False):
        # TODO: to_lower results in posting 0 utterances to training. Why?
        if to_lower:
            questions = questions.apply(lambda x: x.astype(str).str.lower())

        questions['Intent'] = questions['Intent'].astype(str).str.lower()
        length_before = len(questions)

        if questions.apply(lambda x: x.astype(str).str.lower()).duplicated(subset=['Question', 'Intent']).any() == True:
            #duplicates = questions[questions.duplicated(subset=['Question','Intent'])]
            #print('List of duplicate utterances to be disregarded:')
            # print(set(list(duplicates.Question)))
            questions_lower = questions.apply(
                lambda x: x.astype(str).str.lower())
            dup_inds = questions_lower[questions_lower.duplicated(
                subset='Question', keep='first')].index
            questions = questions.drop(index=dup_inds)
            #print('Dataset now formed of distinct Utterance-Intent pairs.')

        if questions[pd.isnull(questions['Intent'])].shape[0] > 0:
            #print('List of utterances with empty intents to be disregarded:')
            #questions_list = list(questions[pd.isnull(questions['Intent'])]['Question'])
            # print(questions_list)
            questions['TestSetRef'] = 'temporary value'
            questions = questions.dropna()
            questions['TestSetRef'] = np.nan
            #print('Dataset now formed by fully-populated Utterance-Intent pairs.')

        if questions[pd.isnull(questions['Question'])].shape[0] > 0:
            #print('Empty utterances found:')
            # print(questions[pd.isnull(questions['Question'])])
            questions = questions.dropna(subset=['Question'])

        elements = list(questions['Question'])
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
                no_tosample = int(samples_per_intent['Question'][intent])

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

    def post_workspace(self, questions, topic, wcsLanguage="en"):
        """
        posts a workspace to the instance specified in credentials 
        returns a list containing workspace ID
        """

        # create temp workspace
        workspaceList = []
        testNode = [{"dialog_node": "kfoldtest", "description": "K-Fold Test Node",
                     "conditions": "context.kfoldTest == true", "output":
                     {"text": {"text": "K-Fold Test Node"}}}]

        if 'TestSetRef_1' not in questions.columns.values:
            # column needed for build_intent_json function
            questions['TestSetRef_1'] = 0
        # DEBUG
        questions.to_csv('data/debug/questions_posted.csv')
        intentList = mc.build_intent_json(questions, 1)
        response = mc.ctk.create_workspace(name="trainingatscale_" + topic,
                                           description="testws_" + str(1),
                                           language=wcsLanguage,
                                           dialog_nodes=testNode,
                                           intents=intentList)
        workspaceList.append(response['workspace_id'])

        # wait until wcs finished training
        while mc.isWCSTraining(workspaceList):
            print('Checking Workspaces status....  In Training')
            time.sleep(10)

        return workspaceList

    def new_train_listener(self, new_train, iteration, last_iteration, output_path='./data/debug/new_train_tracker.csv'):
        """
        Takes new training dataframes, appends them with iteration number, then outputs a dataframe to output_path
        when last_iteration == True
        """
        # TODO: need class structure for this to persist data properly
        pass

    def iteratively_add_test(self, method, orig_workspace_id_list, initial_questions_train, questions_remaining, blindset_questions, no_iterations=3, train_blind_split=0.875):
        """
        Combines the available data leftover from the workspace, and the blindset.
        Randomly (stratified) splits this into set to use for training and a blindset.

        Iteratively:
        - Selects which utterances to prioritise (the method to do this may run a blindset test)
        - Runs a set number of these utterances through a blind set test
        - Exports results
        """

        # Combine all available data, then split it into train and test sets
        available_data = questions_remaining.append(
            blindset_questions, ignore_index=True, sort=False)
        available_train, test_set = self.train_test_split(
            available_data, sampleSize=train_blind_split)
        utterances_per_iteration = int(len(available_train) / no_iterations)

        # Run blind test against existing workspace
        print('Running blind test for ' + str(len(test_set)) + ' utterances...')
        # TODO: use while True and if, break as in https://stackoverflow.com/questions/2244270/get-a-try-statement-to-loop-around-until-correct-value-obtained
        try:
            self.orig_results = bt.runBlindTest(
                orig_workspace_id_list[0], test_set, show_progress=False, get_data=True)
        except:
            try:
                print('retrying (1)')
                self.orig_results = bt.runBlindTest(
                    orig_workspace_id_list[0], test_set, show_progress=False, get_data=True)
            except:
                try:
                    print('retrying (2)')
                    self.orig_results = bt.runBlindTest(
                        orig_workspace_id_list[0], test_set, show_progress=False, get_data=True)
                except:
                    print('blind test failed thrice')


        self.orig_results.to_csv('results/' + method + '_orig.csv')

        # If method requires it, then run available_train against existing corpus to get confidences
        # Use these blindset results instead of available_train
        if method in ['margin', 'lowconf']:
            print('Running available_train (' +
                  str(len(available_train)) + ') against original corpus')
            orig_available_train_bsresults = bt.runBlindTest(
                orig_workspace_id_list[0], available_train, show_progress=False, get_data=True)

            available_train = orig_available_train_bsresults

        # TODO: is continually overwriting this variable the best thing to do?
        current_workspace_list = orig_workspace_id_list
        self.current_workspace_list = current_workspace_list
        self.orig_workspace_id_list = orig_workspace_id_list

        # Iteratively add utterances and test each time
        questions_in_training = initial_questions_train.copy()
        for i in range(0, no_iterations):
            print('--ITERATION ' + str(i+1) + '/' + str(no_iterations) + '--')
            print(str(len(available_train)) + ' utterances available for training. Aiming to get ' +
                  str(utterances_per_iteration) + '.')
            print('Getting new utterances using method ' + method)
            # get new utterances
            new_train = self.get_priority_utterances(available_train, utterances_per_iteration,
                                                     questions_in_training, current_workspace_list, method)
            print('Got ' + str(len(new_train)) + ' utterances')                

            # DEBUG
            available_train.to_csv(
                'data/debug/available_train_' + self.topic + str(i) + '.csv')
            new_train.to_csv('data/debug/new_train_' + self.topic + str(i) + '.csv')

            # TODO: drop new_train utterances from those available
            len_before = len(available_train)
            available_train = available_train.drop(new_train.index)
            len_after = len(available_train)
            print('Dropped ' + str(len_before - len_after) +
                  ' utterances from available_train.')
            # delete temporary workspace, unless it was the original workspace
            if i > 0:
                mc.deleteWorkspaces(current_workspace_list)

            # push amended workspace to Watson
            questions_in_training = pd.concat(
                [questions_in_training, new_train], sort=False)
            questions_in_training = self.check_questions_df_consistency(
                questions_in_training)
            questions_in_training.to_csv(
                'data/debug/questions_in_training_' + self.topic + str(i) + '.csv')
            current_workspace_list = self.post_workspace(
                questions_in_training, topic)
            no_added_to_training = len(new_train)

            # run blind set test and save results
            print('Running blind test for ' +
                  str(len(test_set)) + ' utterances...')
            try:
                results = bt.runBlindTest(
                    current_workspace_list[0], test_set, show_progress=False, get_data=True)
                print('running blind test (1)')
                
            except:
                try:
                    results = bt.runBlindTest(
                        current_workspace_list[0], test_set, show_progress=False, get_data=True)
                    print('retrying blind test')
                except:
                    try:
                        results = bt.runBlindTest(
                            current_workspace_list[0], test_set, show_progress=False, get_data=True)
                        print('retrying blind test (2)')
                    except:
                        print('blind test failed thrice')
            
            results_path = 'results/' +  self.topic + '_' + method + '_it' + \
                str(i + 1) + '_noadded_' + str(no_added_to_training) + '.csv'
            results.to_csv(results_path)
            print('Blind test results saved to ' + results_path)

            if i == (no_iterations - 1):
                mc.deleteWorkspaces(current_workspace_list)

            if len(new_train) < utterances_per_iteration:
                print('Stopping at this iteration as there is not enough data to continue.')
                break

    def get_priority_utterances(self, utterance_list, no_utterances, questions_in_training, current_workspace_list, method):
        """
        Switching function which calls other functions to get priority utterances
        """

        if method == 'random':
            questions = utterance_list.sample(no_utterances)

        elif method == 'lowconf':
            questions = self.utterance_select_lowconf(
                utterance_list, no_utterances, current_workspace_list)

        elif method == 'margin':
            questions = self.utterance_select_margin(
                utterance_list, no_utterances, current_workspace_list)

        elif method == 'similarity':
            questions = self.utterance_select_dissimilarity(
                utterance_list, no_utterances, questions_in_training)

        elif method == 'small_intents':
            questions = self.utterance_select_smaller_intents(
                utterance_list, no_utterances, questions_in_training)

        else:
            print('Method not defined.')

        return questions

    def utterance_select_margin(self, utterance_list, no_utterances, prev_trained_workspace_list):
        """
        Run a blind set test of all the remaining utterances against the current trained workspace.
        Based on these results, select the utterances which have the lowest margin (below a threshold?)
        Return these utterances and their intents in a dataframe
        """
        # TODO: put an upper limit on the margin as a condition for returning utterances?
        results = utterance_list.copy()
        results['margin'] = results['Confidence1'] - results['Confidence2']
        questions = results.sort_values(
            'margin', ascending=True).head(no_utterances)

        # filter using self.lowconf_max
        size_before_filter = len(questions)
        questions = questions[questions['margin'] <= self.margin_max]
        size_after_filter = len(questions)

        if size_after_filter < size_before_filter:
            print('Only ' + str(size_after_filter) + '/' + str(no_utterances) + ' utterances present with a margin below the max of ' + str(self.lowconf_max))


        print(str(no_utterances) + ' utterances returned with an average margin of ' +
              str(questions['margin'].mean()))
        questions = questions[['Question', 'Expected Intent']].rename(
            columns={'Expected Intent': 'Intent'})

        return questions

    def utterance_select_lowconf(self, utterance_list, no_utterances, prev_trained_workspace_list):
        """
        Run a blind set test of all the remaining utterances against the current trained workspace.
        Based on these results, select the utterances which have the lowest margin (below a threshold?)
        Return these utterances and their intents in a dataframe
        """
        # TODO: maybe put an upper limit on the confidence that can be returned here
        results = utterance_list.copy()
        questions = results.sort_values(
            'Confidence1', ascending=True).head(no_utterances)

        # filter using self.lowconf_max
        size_before_filter = len(questions)
        questions = questions[questions['Confidence1'] <= self.lowconf_max]
        size_after_filter = len(questions)

        if size_after_filter < size_before_filter:
            print('Only ' + str(size_after_filter) + '/' + str(no_utterances) + ' utterances present with a confidence@1 below the max of ' + str(self.lowconf_max))
       
        print(str(no_utterances) + ' utterances returned with an average confidence@1 of ' +
            str(questions['Confidence1'].mean()))

        questions = questions[['Question', 'Expected Intent']].rename(
            columns={'Expected Intent': 'Intent'})

        return questions

    def utterance_select_dissimilarity(self, utterance_list, no_utterances, questions_in_training):
        """
        Select no_utterances from utterance_list prioritised by their dissimilarity from the training corpus
        (questions_in_training).
        TODO: This might be a general point: if there aren't enough utterances to return, do we 
        a) lower the similarity threshold 
        b) return fewer utterances than requested, or
        c) return a random amount?
        For now, flagging it and returning all that were similar to at least 1 utterance in training. 
        Should find another method to switch to in future. 
        """
        # get number of utterances in the training set that each available utterances is similar to
        # TODO: is there a better measure of dissimilarity?
        mh = MinHash_similarity(threshold=0.5, num_perm=512, shingle_length=5)
        df_query = mh.similarity_threshold_bulk(
            df_library=questions_in_training, df_query=utterance_list, return_df=True)
        no_available = len(df_query[df_query['no_similar'] > 0])

        if no_utterances < no_available:
            print('Not enough utterances available (' + str(no_utterances) +
                  '). Returned all ' + str(no_available) + ' available utterances.')
            no_utterances = no_available

        return df_query.sort_values('no_similar', ascending=False).head(no_utterances)

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
        return utterance_list.sample(n=no_utterances, weights='norm_prob')[['Question', 'Intent']]


if __name__ == '__main__':
    topic = 'payments'
    topic_thresh = 0.4
    #methods = ['random', 'margin']
    methods = ['similarity', 'lowconf']
    sampleSize = 0.05  # default 0.2
    no_iterations = 7  # default ?
    train_blind_split = 0.8

    """ PARAMS HISTORY
    - travel default
    - payments/products: sampleSize = 0.1
    - cards: sampleSize = 0.05; train_blind_split = 0.8
    """

    margin_max = 0.1
    lowconf_max = 0.5
    # TODO: pull out variables for similarity method
    rt = run_test(margin_max=margin_max, lowconf_max=lowconf_max, topic=topic)

    intentpath = 'data/'+topic+'_questions.csv'
    blindsetpath = 'data/'+topic+'_blindset.csv'
    questions = rt.import_data(intentpath)

    if len(questions) > 1000:
        questions = questions.sample(n=1000).reset_index(drop=True)
        print('Sampled 1000 items from original training set.')

    blindset = rt.import_data(blindsetpath)
    #blindset = blindset.iloc[1:30]  # DEBUG

    intial_questions_train, initial_questions_test = rt.train_test_split(
        questions, sampleSize)

    workspace_list = rt.post_workspace(intial_questions_train, topic)
    print(workspace_list)

    for method in methods:
        rt.iteratively_add_test(method, workspace_list, intial_questions_train,
                                initial_questions_test, blindset, no_iterations, train_blind_split=train_blind_split)

        summarise = summarise_results.summarise_results(
            workspace_conf_thresh=topic_thresh, method=method, topic=topic)
        joined_df = summarise.join_results_individual_method(method)
        confmat_orig = summarise.calculate_stats_per_intent(rt.orig_results)
        confmat_orig.to_csv('results/confmat_orig_' + topic + '_' + method + '.csv')
        confmat = summarise.calculate_stats_per_intent_all_iterations(
            joined_df, export_csv=True)
        
    # TODO: is this in the right place?
    mc.deleteWorkspaces(rt.current_workspace_list)
