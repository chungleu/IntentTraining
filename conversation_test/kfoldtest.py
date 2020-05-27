import json 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from ibm_watson import AssistantV1
from IPython.display import display
from sklearn.metrics import *
from sklearn.model_selection import *
import itertools
from tqdm import tqdm 
import time

import os
import sys
sys.path.append('..')

import for_csv.logger
from logging import getLogger
logger = getLogger("kfoldtest")
import config
import click

# internal imports
from conversation_test.metrics import Metrics

@click.command()
@click.argument('topic', nargs=1)
@click.option('--no_folds', '-n', type=int, default=5, help="No. folds to run for kfold test")
@click.option('--results_type', '-r', type=click.Choice(['raw', 'metrics_intent', 'all']), default='all', help='Whether to give raw results per utterance, metrics (per intent), or all available.')
@click.option('--conf_matrix', '-c', is_flag=True ,help='Whether to plot a confusion matrix.')
def run_kfold(topic, no_folds, results_type, conf_matrix):
    """
    Runs kfold test using credentials in ../Credentials.py
    """

    # get credentials, import + export folders
    import Credentials
    active_adoption = Credentials.active_adoption
    instance_creds = Credentials.ctx[active_adoption]
    workspace_id = Credentials.workspace_id[active_adoption][topic]
    workspace_thresh = Credentials.calculate_workspace_thresh(topic)
    conversation_version = Credentials.conversation_version

    # import + export folders
    import config
    import time
    data_folder = config.data_dir
    export_folder = config.output_folder
    timestr = time.strftime("%Y%m%d-%H%M")
    
    output_loc_results = os.path.join(export_folder, "{}_kfold_results_raw_{}.csv".format(topic, timestr))
    output_loc_metrics = os.path.join(export_folder, "{}_kfold_results_metrics_{}.csv".format(topic, timestr))
    output_loc_confmat = os.path.join(export_folder, "{}_kfold_confmat_{}.png".format(topic, timestr))

    # authenticate
    if 'apikey' in instance_creds:
        logger.debug("Authenticating (apikey)")
        kf = kfoldtest(n_folds=no_folds, apikey=instance_creds['apikey'], url=instance_creds['url'], threshold=workspace_thresh, version = conversation_version)
    elif 'password' in instance_creds:
        logger.debug("Authenticating (username/password)")
        kf = kfoldtest(n_folds=no_folds, username=instance_creds['username'], password=instance_creds['password'], url=instance_creds['url'], threshold=workspace_thresh, 
            version=conversation_version)

    # get train df from watson + check there are sufficient workspaces to run the test
    train_df = kf.intent_df_from_watson(workspace_id)
    kf.check_sufficient_workspaces()

    # create folds in WA if above is true
    folds = kf.create_folds(method='kfold')
    kf.create_kfold_WA(folds)

    available_flag = False

    while available_flag == False:
        logger.info("Checking workspaces..")
        available_flag = kf.check_workspaces_status()
        time.sleep(20)

    # run kfold test 
    try: 
        results = kf.run_kfold_test(folds)

        if (results_type == 'raw') or (results_type == 'all'):
            results.to_csv(output_loc_results)

        classification_report = kf.create_classification_report(results)

        if (results_type == 'metrics') or (results_type == 'all'):
            metrics = Metrics(workspace_thresh)
            metric_df = metrics.get_all_metrics_CV(results, fold_col='fold', detailed_results=False)
            metric_df.to_csv(output_loc_metrics)

        # TODO: confusion matrix
        if conf_matrix:
            from confusionmatrix import ConfusionMatrix
            cfn = ConfusionMatrix(workspace_thresh=workspace_thresh)
            cfn.create(results, fig_path=output_loc_confmat)
            logger.info("Confusion matrix saved to {}".format(output_loc_confmat))

    finally:
        # regardless of what happens above, delete the temporary workspaces before exiting
        kf.delete_kfold_workspaces()

class kfoldtest(object):
    """
    One object per Watson Assistant instance. Blindset test can be run for different workspace IDs or test files using the 
    run_blindset_test method. 
    ARGS:
    - username & password, or apikey
    - url
    - threshold
    - n_folds
    """
    def __init__(self, **kwargs):
        
        if 'url' in kwargs:
            self.url = kwargs['url']
        else:
            raise ValueError("URL needs to be provided.")

        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']
        else:
            raise ValueError("Threshold needs to be provided")

        if 'n_folds' in kwargs:
            self.n_folds = kwargs['n_folds']
        else:
            self.n_folds = 5
            logger.warn("Number of folds not provided so set to 5 as default.")
        
        # make sure all variables are here
        if ('apikey' not in kwargs) and (any(('username', 'password')) not in kwargs):
            raise ValueError("One of username & password, or apikey must be present. ")

        if 'apikey' in kwargs:
            self.apikey = kwargs['apikey']
            self.auth_type = 'apikey'

        if ('username' in kwargs) and ('password' in kwargs):
            self.username = kwargs['username']
            self.password = kwargs['password']
            self.auth_type = 'password'

        if 'version' in kwargs:
            self.conversation_version = kwargs['version']
        else:
            self.conversation_version = '2018-07-10'

        # authenticate
        self.authenticate_watson()

        # list of workspaces that the instance has created
        self.workspaces = []

        # TODO: get max number of workspaces in instance from API call

    def authenticate_watson(self):
        """
        Connect to WA and return an assistant instance. 
        Takes apikey or username & password kwargs depending on auth_type.
        """

        if self.auth_type == 'apikey':
            assistant = AssistantV1(
                iam_apikey=self.apikey,
                version=self.conversation_version, 
                url= self.url)

        elif self.auth_type == 'password':
            assistant = AssistantV1(
                username=self.username,
                password=self.password,
                version=self.conversation_version,
                url= self.url)  

        self.assistant = assistant


    def intent_df_from_json(self, workspace_json_path):
        """
        Reads a workspace JSON, returns a dataframe with intents and examples.
        """
        
        with open(workspace_json_path) as f: # USER INPUT - CHANGE THE FILE NAME HERE
            data = json.load(f)

       
        df = pd.DataFrame(columns = ['intent', 'utterance'])
        
        for i in range(len(data['intents'])):
            #print("Scanned intent: {}".format(data['intents'][i]['intent']))
            for j in range(len(data['intents'][i]['examples'])):
                df = df.append({'intent': data['intents'][i]['intent'],
                            'utterance': data['intents'][i]['examples'][j]['utterance']}
                        ,ignore_index=True)

        self.training_df = df

        return df

    def intent_df_from_watson(self, workspace_id):
        """
        Reads the workspace via API returns a dataframe with intents and examples.
        """
        # Call WA to ge the list of the intents 
        response = self.assistant.list_intents(workspace_id = workspace_id, page_limit=10000)
        obj = json.dumps(response.get_result(), indent=2)
        data = json.loads(obj)
        
        df = pd.DataFrame(columns = ['intent','utterance'])
        
        for i in range(len(data["intents"])): 
            name_intent = data["intents"][i]["intent"]

            # Call WA to get the list of Examples of each intent 
            response = self.assistant.list_examples(workspace_id = workspace_id, intent = name_intent, page_limit=10000)
            dumps = json.dumps(response.get_result(), indent=2)
            data_examples = json.loads(dumps)

            # get the Groud Truth (examples test) of each intent 
            for j in range(len(data_examples["examples"])): 
                text = data_examples["examples"][j]["text"]
                df = df.append({'intent':name_intent,'utterance': text},ignore_index=True)
            
            #print ("Scanned intent: " , name_intent )
        
        self.training_df = df

        return df 

    def intent_df_from_df(self, df):
        """
        Use an existing df as train_df
        """

        self.training_df = df

    def create_folds(self, method, sample_size=0.2):
        """
        Create folds either through stratified kfold or monte-carlo.
        Monte-Carlo is always stratified. 
        :param method: one of [kfold, monte-carlo].
        :param sample_size: the proportion of utterances to hold out for monte-carlo testing.
        :return folds: a list of folds containing the train and test indices for each fold
        """

        if method == "kfold":
            return self.create_folds_kfold()
        elif method == "monte-carlo":
            return self.create_folds_mc(sample_size)
        else:
            raise ValueError("Parameter method must either be 'kfold' or 'monte-carlo'")

    def create_folds_kfold(self):
        """
        Create the folds for the k-fold test. It is using the Stratified K-fold division. 
        
        :param self.training_df: the dataframe containing the whole GT of the workspace 
        :return folds: a list of folds containing for each fold the train and test set indexes. 
        """

        folds = []
        i = 0
        skf = StratifiedKFold(n_splits = self.n_folds, shuffle = True, random_state = 2)
        for train_index, test_index in skf.split(self.training_df['utterance'], self.training_df['intent']):
            fold = {"train": train_index,
                    "test": test_index}
            folds.append(fold)
            logger.debug("fold num {}: train set: {}, test set: {}".format(i+1,len(fold["train"]), len(fold["test"])))
            i += 1
        
        return folds

    def create_folds_mc(self, sample_size):
        """
        Create folds for a Monte-Carlo test. Each fold is a new and independent random sample, stratified 
        with respect to intent size.
        :param: proportion of training set held out as test set.
        """

        folds = []
        samples_per_intent = np.ceil(self.training_df.groupby('intent').count() * sample_size)
        intents = self.training_df['intent'].unique().tolist()

        for i in range(0,self.n_folds):
            train_inds = []
            test_inds = []
            
            for intent in intents:
                questions_intent = self.training_df[self.training_df['intent'] == intent]
                no_tosample = int(samples_per_intent['utterance'][intent])
                test_inds += questions_intent.sample(n=no_tosample).index.tolist()
                train_inds += list(set(questions_intent.index.tolist()) - set(test_inds))

            fold = {"train": train_inds,
                    "test": test_inds}
            folds.append(fold)
            logger.debug("fold num {}: train set: {}, test set: {}".format(i+1,len(fold["train"]), len(fold["test"])))

        return folds

    def check_sufficient_workspaces(self):
        """
        counting the existing workspaces and check if there is space for k-workspaces 
        """
        response = self.assistant.list_workspaces().get_result()
        k_fold_number = self.n_folds
        max_workspaces = config.max_workspaces

        if(len(response['workspaces'])+k_fold_number <= max_workspaces):
            logger.info("You have space to perform the k-fold test")
        else: 
            remove = len(response['workspaces'])+k_fold_number-max_workspaces
            raise ValueError("The K-fold test will make you exceed the {}} workspaces limit. Make "
            "sure to remove {} workspaces before creating the k-fold workspaces".format(remove, max_workspaces))
        

    def create_intents(self, train_index):
        """
        It collects the intents in json format to send when creating the workspace 
            
        :param train_index: that are the results of the 'create_folds' function
        :return intent_results: if a list of dictionaries that will be sent when new workspace will be created
        """
        
        intent_results = []
        for i in train_index:
            row = {}
            text = self.training_df.iloc[i]['utterance']
            intent = self.training_df.iloc[i]['intent']

            if not any(d['intent'] == intent for d in intent_results):
                row = { 'intent': intent, 
                        'examples': [ {'text': text } ] } 
            else:
                row = [d for d in intent_results if d.get('intent') == intent][0]
                intent_results[:] = [d for d in intent_results if d.get('intent') != intent]
                e = {'text': text}
                row['examples'].append(e)

            intent_results.append(row)
        
        return intent_results

    def create_workspace(self, intents_json, fold_number):
        """
        create one workspace 
        
        :param intent_json : output of the 'create_intents' function
        :param fold_number: the number of the fold  
        :return workspace_id: the id of the workspace that has been generated
        """
        response = self.assistant.create_workspace(
            name='K_FOLD test {}'.format(fold_number+1),
            #language = 'en'   # CHANGE LANGUAGE HERE (Default is 'en')
            description='workspace created for k-fold testing', 
            intents = intents_json
        ).get_result()
        
        workspace_id = response.get('workspace_id')
        
        return workspace_id

    def create_kfold_WA(self, folds):
        """
        create the k-fold workspaces in WA
        
        :param folds: are the folds created in the function `create_folds`
        :return workspaces: is a list of workspaces ID generated 
        """
        logger.info("Creating kfold workspaces..")
        
        for i in range(len(folds)):
            train = folds[i]["train"]
            intents = self.create_intents(train)
            workspace_id = self.create_workspace(intents, i)
            self.workspaces.append(workspace_id)
        
        return self.workspaces

    def check_workspaces_status(self): 
        """
        check the status of the workspace just created - You can start the k-fold only when 
        the workspaces are `Available` and not in Training mode. 

        Returns available flag to be used inside a while loop
        """

        available_count = 0

        workspaces = self.workspaces

        for i in range(len(workspaces)):
            response = self.assistant.get_workspace(workspace_id = workspaces[i]).get_result()
            status = response['status']
            # The status can be: unavailable, training, non-existent, failed 
            if (status == 'Available'):
                available_count +=1
                logger.debug("Workspace {} available".format(i+1))
        
        return available_count == len(workspaces)


    def test_kfold(self, df_test, ws_id, export_interim=False):
        """
        This function will take the regression test uploaded in csv and will send each phrase to WA and collect 
        information on how the system responded. 
        
        :param df_test: the dataframe containing the testing phrases 
        :param ws-id: the index of the fold that would be used to call the correct workspace id that needs to be test 
        :return results: a pandas dataframe with original text, predicted intent and also the results from WA
        """
        results = pd.DataFrame([],columns = ['original_text','expected intent','r@1','TP','intent1',
                            'confidence1','intent2','confidence2','intent3',
                            'confidence3'])

        for i in tqdm(range(len(df_test))):

            text = df_test['utterance'][i]

            response = self.assistant.message(workspace_id=self.workspaces[ws_id], input={'text': text}, alternate_intents= True)
            dumps = json.dumps(response.get_result(), indent=2)
            
            data = json.loads(dumps)

            no_intents = len(data['intents'])

            intent1= data['intents'][0]['intent']
            confidence1 = data['intents'][0]['confidence']

            if no_intents >= 2:
                intent2= data['intents'][1]['intent'] 
                confidence2 = data['intents'][1]['confidence'] 
            else:
                intent2 = confidence2 = ""

            if no_intents >= 3:
                intent3= data['intents'][2]['intent']
                confidence3 = data['intents'][2]['confidence']
            else:
                intent3 = confidence3 = ""

            r_1 = df_test["intent"][i] == intent1
            tp = r_1 and (confidence1 >= self.threshold)

            results = results.append({
                    'original_text': df_test["utterance"][i],
                    'expected intent': df_test["intent"][i],
                    'r@1': 1*r_1,
                    'TP': 1*tp,
                    'intent1': intent1, 
                    'confidence1':confidence1, 
                    'intent2':intent2, 
                    'confidence2': confidence2, 
                    'intent3': intent3,
                    'confidence3': confidence3, 
                }, ignore_index=True)
            
        if export_interim:
            results.to_csv("./kfold_{}_raw.csv".format(ws_id+1), encoding='utf-8')
        
        return results

    def run_kfold_test(self, folds):
        """
        run the k-fold test. It is going to take folds as input and it will send the test dataframes to the right
        workspaces. 
        
        :param folds: output list from the function `create_folds`
        :return test_results: is list of results (dataframes) for each fold.  
        """
        test_results = pd.DataFrame()
        for i in range(len(folds)):
            logger.info("Running test for fold {}".format(i+1))
            test_index = folds[i]['test']
            df_test = self.training_df.iloc[test_index]
            df_test_reindexed = df_test.reset_index()
            results = self.test_kfold(df_test_reindexed, i)
            results["fold"] = i+1
            test_results = test_results.append(results)

        test_results.loc[:, "intent_correct"] = test_results["intent1"]
        test_results["intent_correct"] = np.where((test_results["confidence1"]<self.threshold), "BELOW_THRESHOLD", test_results["intent1"])

        return test_results

    def get_metrics_per_fold(self, results_kfold):
        """
        Calculates metrics per fold
        
        :param results_kfold: is the list of results coming from `run_kfold_test` function
        :return result_table: is the dataframe containing the metrics for each fold. 
        """
        result_table = pd.DataFrame([],columns=["fold","total_tested","incorrect","accuracy", "precision","recall","fscore"])

        for i in range(len(results_kfold)):
            data = results_kfold[results_kfold['fold'] == i+1]
            incorrect_n = data.loc[data['intent_correct']!=data["expected intent"]]
            incorrect_avg_conf = incorrect_n['confidence1'].mean()
            precision,recall,fscore,support=precision_recall_fscore_support(data["intent_correct"],data["expected intent"],average='weighted')
            accuracy = accuracy_score(data["intent_correct"], data["expected intent"])
            result_table = result_table.append({
                "fold": i+1,
                "total_tested": len(data),
                "incorrect": len(incorrect_n),
                "incorrect_avg_confidence": incorrect_avg_conf,
                "accuracy": accuracy, 
                "precision": precision, 
                "recall": recall, 
                "fscore": fscore
            }, ignore_index=True)
        
        return result_table

    def data_prep_confusion_matrix(self, list_df):
        """
        this function prepares the dataframe to be then used for the confusion matrix 
        
        :param list_df: is the list of dataframes (results) coming from each fold. 
        :return matrix: it is the confusion matrix that will be displayed in `plot_confusion_matrix`
        :return lab: the lables that are used for the visualisation 
        """
         
        list_df
        
        list_df["intent_correct"] = list_df["intent1"]
        list_df["intent_correct"] = np.where((list_df["confidence1"]<self.threshold), "BELOW_THRESHOLD", list_df["intent1"])
        matrix = confusion_matrix(list_df["intent_correct"], list_df["expected intent"])
        
        lab1 = list_df["intent_correct"].unique()
        lab2 = list_df["expected intent"].unique()
        lab = np.union1d(lab1,lab2)
        
        return matrix, lab, list_df

    def plot_confusion_matrix(self, cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.RdPu):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix for the Intent matching")
        else:
            print('Confusion matrix for the Intent matching')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Actual Intent')
        plt.xlabel('Predicted Intent')
        plt.tight_layout()
        
        return 
    
    def calculate_overall_metrics(self, results, av_method='weighted'):
        """
        Gets results for whole corpus tested from a results file. 
        """
        accuracy = accuracy_score(results["intent_correct"], results["expected intent"])

        precision,recall,fscore,support=precision_recall_fscore_support(results["intent_correct"],
                                                                        results["expected intent"],
                                                                        average=av_method)

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'F1Score': fscore}

    def create_classification_report(self, results):
        """
        Returns a dataframe showing precision, recall and F1 for each intent and overall for the workspace.
        """
        
        report = classification_report(results["intent_correct"], results["expected intent"], output_dict=True)
        report_df = pd.DataFrame.from_dict(report)
        report_df = report_df.drop(columns = ["BELOW_THRESHOLD"])
        
        # TODO: not sure where this col has come from
        if "accuracy" in report_df.columns:
            report_df = report_df.drop(columns = 'accuracy')
        
        return report_df.T    

    def delete_kfold_workspaces(self):
        """
        delete the workspaces when you dont need them anymore
        """

        workspaces = self.workspaces

        logger.info("Deleting temporary workspaces")

        for i in range(len(workspaces)):
            response = self.assistant.delete_workspace(
                    workspace_id = workspaces[i]).get_result() 

        self.workspaces = []

    def full_run_kfold_from_df(self, train_df, method='kfold'):
        """
        Given a training df, will run kfold tests and output all metrics.
        """
        self.check_sufficient_workspaces()
        folds = self.create_folds(method)
        self.create_kfold_WA(folds)

        available_flag = False
        
        while available_flag == False:
            logger.info("Checking workspaces..")
            available_flag = self.check_workspaces_status()
            time.sleep(20)
        
        try:
            # results per utterance
            results_kfold = self.run_kfold_test(folds)

            # metrics per intent
            metrics = Metrics(self.threshold)
            classification_report = metrics.get_all_metrics_CV(results_kfold, fold_col='fold', detailed_results=False)

        finally:
            self.delete_kfold_workspaces()

        return results_kfold, classification_report

if __name__ == "__main__":
    run_kfold()
    
