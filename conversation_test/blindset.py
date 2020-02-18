# ## Libraries
import json 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from ibm_watson import AssistantV1
from IPython.display import display
from sklearn.metrics import *
import itertools
import click
import os
import sys
sys.path.append('..')

import for_csv.logger
from logging import getLogger
logger = getLogger("blindset")

from conversation_test.metrics import Metrics

@click.command()
@click.argument('topic', nargs=1)
@click.option('--results_type', '-r', type=click.Choice(['raw', 'metrics', 'all']), default='all', help='Whether to give raw results per utterance, metrics, or both.')
@click.option('--conf_matrix', '-c', is_flag=True ,help='Whether to plot a confusion matrix.')
@click.option('--blindset_name', '-n', default=None, help='Specify which csv file in the data folder to use as a test set. Just have to give the relative path from the data folder (the filename if the file is in the data folder).')
def run_blindset(topic, results_type, conf_matrix, blindset_name):
    """
    Runs blindset test using credentials in ../Credentials.py
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
    
    blindset_name = blindset_name or topic + "_blindset.csv"
    output_loc_results = os.path.join(export_folder, "{}_results_raw_{}.csv".format(topic, timestr))
    output_loc_metrics = os.path.join(export_folder, "{}_results_metrics_{}.csv".format(topic, timestr))
    output_loc_confmat = os.path.join(export_folder, "{}_confmat_{}.png".format(topic, timestr))

    # authenticate
    if 'apikey' in instance_creds:
        logger.debug("Authenticating (apikey)")
        bs = blindset(apikey=instance_creds['apikey'], url=instance_creds['url'], threshold=workspace_thresh, version=conversation_version)
    elif 'password' in instance_creds:
        logger.debug("Authenticating (username/password)")
        bs = blindset(username=instance_creds['username'], password=instance_creds['password'], url=instance_creds['url'], threshold=workspace_thresh, 
            version=conversation_version)
    
    # run test
    blindset_df = bs.import_blindset(os.path.join(data_folder, blindset_name))
    # TODO: check blindset df
    results = bs.run_blind_test(blindset_df, workspace_id)

    # exports + metrics
    if (results_type == 'raw') or (results_type == 'all'):
        cols_export = [col for col in results.columns.values if col != 'intent_correct']
        results[cols_export].to_csv(output_loc_results, encoding='utf-8')
        logger.info("Raw results exported to {}".format(output_loc_results))

    if (results_type == 'metrics') or (results_type == 'all'):
        met = Metrics(workspace_thresh)
        metric_df, _ = met.get_all_metrics(results, detailed_results=True)

        metric_df.to_csv(output_loc_metrics, encoding='utf-8')
        logger.info("Metrics per intent exported to {}".format(output_loc_metrics))

    # confusion matrix
    if conf_matrix:
        from confusionmatrix import ConfusionMatrix
        cfn = ConfusionMatrix(workspace_thresh=workspace_thresh)
        cfn.create(results, fig_path=output_loc_confmat)
        #bs.plot_confusion_matrix(results, output_loc_confmat)    
        logger.info("Confusion matrix saved to {}".format(output_loc_confmat))

    # print high-level metrics
    overall_metrics = bs.calculate_overall_metrics(results, av_method="weighted")
    logger.info("Overall metrics for the workspace (weighted):")
    logger.info(overall_metrics)
    
    # TODO: check consistency of test set before running.


class blindset(object):
    """
    One object per Watson Assistant instance. Blindset test can be run for different workspace IDs or test files using the 
    run_blindset_test method. 
    ARGS:
    - username & password, or apikey
    - url
    - threshold
    - version
    """
    def __init__(self, **kwargs):
        
        if 'url' in kwargs:
            self.url = kwargs['url']
        else:
            raise ValueError("URL needs to be provided.")

        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']
        else:
            self.threshold = False
            logger.debug('No threshold provided. Provide one when running the blindset test.')
        
        # make sure all variables are here
        if ('apikey' not in kwargs) and (('username' not in kwargs) or ('password' not in kwargs)):
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

    def import_blindset(self, blindset_path):
        """
        Imports blindset csv as dataframe, ensuring it has the correct headers.
        Returns blindset dataframe. 
        """
        if not blindset_path.endswith('.csv'):
            raise ValueError("Blindset file must be a CSV.")

        test_set_df = pd.read_csv(blindset_path, names=['utterance', 'expected intent'])

        # remove leading and trailing whitespace from intent labels
        test_set_df['expected intent'] = test_set_df['expected intent'].astype(str).str.strip()
        return test_set_df

    def run_blind_test(self, test_set_df, workspace_id, **kwargs):
        """
        Runs blind set test and returns results df.
        
        Parameter: 
            test_set_df: the regression_test in csv format

        Return: 
            results: a Pandas dataframe with `original text`, `predicted intent` and also the results from WA
        """

        # if no threshold has been passed into the object, take one from the function args
        if self.threshold == False and 'threshold' not in kwargs:
            raise ValueError("Must provide a threshold either to the blindset object or this function.")
        elif 'threshold' in kwargs:
            # threshold in function args overwrites one provided to the object, even if one has been set
            threshold = kwargs['threshold']
        else:
            threshold = self.threshold
            
        results = pd.DataFrame(columns=['original_text','expected intent','r@1','intent1','confidence1',
                                        'intent2','confidence2','intent3','confidence3'])
        logger.info("Running blind test...")
        for i in tqdm(range(len(test_set_df))):

            text = test_set_df["utterance"][i]
            response = self.assistant.message(workspace_id=workspace_id, input={'text': text}, alternate_intents= True)
            dumps = json.dumps(response.get_result(), indent=2)

            data = json.loads(dumps)

            intent1= data['intents'][0]['intent']
            intent2= data['intents'][1]['intent']
            intent3= data['intents'][2]['intent']
            confidence1 = data['intents'][0]['confidence']
            confidence2 = data['intents'][1]['confidence']
            confidence3 = data['intents'][2]['confidence']
            r_1 = 1*(test_set_df["expected intent"][i] == intent1)
            results = results.append({
                'original_text': test_set_df["utterance"][i],\
                'expected intent': test_set_df["expected intent"][i],\
                'r@1':  r_1, \
                'intent1': intent1, \
                'confidence1':confidence1, \
                'intent2':intent2, \
                'confidence2': confidence2, \
                'intent3': intent3,
                'confidence3': confidence3, \
            }, ignore_index=True)

        results["intent_correct"] = results["intent1"]
        results["intent_correct"] = np.where((results["confidence1"]<self.threshold), "BELOW_THRESHOLD", results["intent1"])

        return results

    def data_prep(self, dataframe):
        """
        this function prepares the dataframe to be used for plot confusion matrix
        """

        matrix = confusion_matrix(dataframe["intent_correct"], dataframe["expected intent"])
        
        lab1 = dataframe["intent_correct"].unique()
        lab2 = dataframe["expected intent"].unique()
        lab = np.union1d(lab1,lab2)
        
        return matrix, lab

    def plot_confusion_matrix(self, results, 
                            save_path=None,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.RdPu):
        """
        This function prints and plots the confusion matrix.
        It also saves it to save_path if specified.
        Normalization can be applied by setting `normalize=True`.
        """

        cm, classes = self.data_prep(results)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12,12))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
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
        
        if save_path:
            plt.savefig(save_path) 

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
        
        return report_df.T


if __name__ == "__main__":
    """ TESTING """
    """     # params
    auth_type = 'apikey'
    apikey = ''
    url = 'https://gateway.watsonplatform.net/assistant/api'
    workspace_id = ''
    test_set_path = '../data/puppy_blindset.csv'
    threshold = 0.4
    results_path = './blindset_results.csv'

    # -- RUN --
    # auth & import
    blindset = blindset(apikey=apikey, url=url, threshold=threshold)
    
    # run blind set test
    test_set_df = blindset.import_blindset(test_set_path)
    results = blindset.run_blind_test(test_set_df, workspace_id, results_path)
    results.to_csv('results.csv')
    # confusion matrix
    blindset.plot_confusion_matrix(results)    
    
    # accuracy score
    overall_results = blindset.calculate_overall_metrics(results, av_method="weighted")
    print("Overall results: {}".format(overall_results))
    
    # classification report
    classification_report_df = blindset.create_classification_report(results)
    classification_report_df.to_csv('classification_report.csv') """
    run_blindset()