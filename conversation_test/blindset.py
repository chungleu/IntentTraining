#!/usr/bin/env python
# coding: utf-8

# # Performance Testing for Watson Assistant 
# This notebook can be used for Regression testing and Blind testing. 
# 
# 1. **Regression Test Scope**: This can be used to test a set of testing phrases that are not part of the training set in Watson Assistant (WA). The idea is that you would re-run this test over time after the improvement phase, to check the health status of your workspace, and make sure that the workspace is still behaving in a consistent manner. 
# 2. **Blind Test Scope**: Simply analyse a set of testing phrases that are not part of the training set in Watson Assistant. This can be a one-off task requested by stakeholders. 
# 
# Either way, the procedure to obtain the results is the same in both cases. The difference is the scope and the content of your testing set.  
# 
# <div class="alert alert-block alert-info">
# <b>Notebook Summary</b>
# <br>
#       
# 1. <b>Connect to WA</b> : credentials to connect to the right WA workspace<br>
# 2. <b>Feed your test set</b> : feed your regression test in a .csv format<br>
# 3. <b>Run the blind test</b> : send the testing phrases to WA workspace <br>
# 4. <b>Analyse the results</b> : calculate the metrics and confusion matrix<br>
# 5. <b>Analyse the incorrect matches</b> : highlight the phrases that did not trigger the right intent  
# </div>
# 

# ## Libraries
import json 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from watson_developer_cloud import AssistantV1
from IPython.display import display
from sklearn.metrics import *
import itertools

class blindset(object):
    """
    One object per Watson Assistant instance. Blindset test can be run for different workspace IDs or test files using the 
    run_blindset_test method. 
    ARGS:
    - username & password, or apikey
    - url
    - threshold
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
                version='2018-07-10', 
                url= self.url)

        elif self.auth_type == 'password':
            assistant = AssistantV1(
                username=self.username,
                password=self.password,
                version='2018-07-10',
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

        return test_set_df

    def run_blind_test(self, test_set_df, workspace_id, results_path):
        """
        Runs blind set test and exports results to CSV.
        
        Parameter: 
            test_set_df: the regression_test in csv format

        Return: 
            results: a Pandas dataframe with `original text`, `predicted intent` and also the results from WA
        """

        results = pd.DataFrame(columns=['original_text','expected intent','intent1','confidence1',
                                        'intent2','confidence2','intent3','confidence3'])
        print("=== BLIND TEST STARTING ===")
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

            results = results.append({
                'original_text': test_set_df["utterance"][i],\
                'expected intent': test_set_df["expected intent"][i],\
                'intent1': intent1, \
                'confidence1':confidence1, \
                'intent2':intent2, \
                'confidence2': confidence2, \
                'intent3': intent3,
                'confidence3': confidence3, \
            }, ignore_index=True)
        
        results.to_csv(results_path, encoding='utf-8', index=False)
        
        print("=== BLIND TEST FINISHED===")
        
        return results

    def data_prep(self, dataframe):
        """
        this function prepares the dataframe to be used for plot confusion matrix
        """
        dataframe["intent_correct"] = dataframe["intent1"]
        dataframe["intent_correct"] = np.where((dataframe["confidence1"]<self.threshold), "BELOW_THRESHOLD", dataframe["intent1"])
        matrix = confusion_matrix(dataframe["intent_correct"], dataframe["expected intent"])
        
        lab1 = dataframe["intent_correct"].unique()
        lab2 = dataframe["expected intent"].unique()
        lab = np.union1d(lab1,lab2)
        
        return matrix, lab

    def plot_confusion_matrix(self, results,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.RdPu):
        """
        This function prints and plots the confusion matrix.
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
        #plt.tight_layout()
        
        plt.savefig('confusion_matrix.png') 

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
    # params
    auth_type = 'apikey'
    apikey = '5XuD4BoYqV0Lx5PUltqzVnDue1MzCTgGcld8uCtPsNR_'
    url = 'https://gateway.watsonplatform.net/assistant/api'
    workspace_id = 'e6b17f68-9a81-4ac6-ae18-231717a47d3a'
    test_set_path = '../data/puppy_blindset.csv'
    threshold = 0.4
    results_path = './blindset_results.csv'

    # -- RUN --
    # auth & import
    blindset = blindset(apikey=apikey, url=url, threshold=threshold)
    
    # run blind set test
    test_set_df = blindset.import_blindset(test_set_path)
    results = blindset.run_blind_test(test_set_df, workspace_id, results_path)
    
    # confusion matrix
    blindset.plot_confusion_matrix(results)    
    
    # accuracy score
    overall_results = blindset.calculate_overall_metrics(results, av_method="weighted")
    print("Overall results: {}".format(overall_results))
    
    # classification report
    classification_report_df = blindset.create_classification_report(results)
    classification_report_df.to_csv('classification_report.csv')