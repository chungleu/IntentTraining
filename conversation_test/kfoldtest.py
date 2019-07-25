import json 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from watson_developer_cloud import AssistantV1
from IPython.display import display
from sklearn.metrics import *
from sklearn.model_selection import *
import itertools
from tqdm import tqdm 
import time

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
            print("Number of folds not provided so set to 5 as default.")
        
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


    def intent_df_from_json(self, workspace_json_path):
        """
        Reads a workspace JSON, returns a dataframe with intents and examples.
        """
        
        with open(workspace_json_path) as f: # USER INPUT - CHANGE THE FILE NAME HERE
            data = json.load(f)

       
        df = pd.DataFrame(columns = ['intent', 'utterance'])
        
        for i in range(len(data['intents'])):
            print("Scanned intent: {}".format(data['intents'][i]['intent']))
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
        response = self.assistant.list_intents(workspace_id = workspace_id, )
        obj = json.dumps(response.get_result(), indent=2)
        data = json.loads(obj)
        
        df = pd.DataFrame(columns = ['intent','utterance'])
        
        for i in range(len(data["intents"])): 
            name_intent = data["intents"][i]["intent"]

            # Call WA to get the list of Examples of each intent 
            response = self.assistant.list_examples(workspace_id = workspace_id, intent = name_intent)
            dumps = json.dumps(response.get_result(), indent=2)
            data_examples = json.loads(dumps)

            # get the Groud Truth (examples test) of each intent 
            for j in range(len(data_examples["examples"])): 
                text = data_examples["examples"][j]["text"]
                df = df.append({'intent':name_intent,'utterance': text},ignore_index=True)
            
            print ("Scanned intent: " , name_intent )
        
        self.training_df = df

        return df 

    def create_folds(self):
        """
        create the folds for the k-fold test. It is using the Stratifies K-fold division. 
        
        :param self.training_df: the dataframe containing the whole GT of the workspace 
        :return folds: a list of folds containing for each fold the train and test set indexes. 
        """

        df = self.training_df
        k_fold_number = self.n_folds

        folds = []
        i = 0
        skf = StratifiedKFold(n_splits = k_fold_number, shuffle = True, random_state = 2)
        for train_index, test_index in skf.split(df['utterance'], df['intent']):
            fold = {"train": train_index,
                    "test": test_index}
            folds.append(fold)
            print("fold num {}: train set: {}, test set: {}".format(i+1,len(folds[i]["train"]), len(folds[i]["test"])))
            i += 1
        
        return folds

    def check_sufficient_workspaces(self):
        """
        counting the existing workspaces and check if there is space for k-workspaces 
        """
        response = self.assistant.list_workspaces().get_result()
        
        if(len(response['workspaces'])+k_fold_number <=20):
            print("You have space to perform the k-fold test")
        else: 
            remove = len(response['workspaces'])+k_fold_number-20
            raise ValueError("Be careful! The K-fold test will make you exceed the 20 workspaces limit. Make "
            "sure to remove {} workspaces before creating the k-fold workspaces".format(remove))
        

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
        workspaces = []
        for i in range(len(folds)):
            print("creating K-FOLD workspace {} out of {}".format(i+1, len(folds)))
            train = folds[i]["train"]
            intents = self.create_intents(train)
            workspace_id = self.create_workspace(intents, i)
            workspaces.append(workspace_id)
        
        return workspaces

    def check_status(self, workspaces): 
        """
        check the status of the workspace just created - You can start the k-fold only when 
        the workspaces are `Available` and not in Training mode. 

        Returns available flag to be used inside a while loop
        """

        available_count = 0

        for i in range(len(workspaces)):
            response = self.assistant.get_workspace(workspace_id = workspaces[i]).get_result()
            status = response['status']
            # The status can be: unavailable, training, non-existent, failed 
            if (status == 'Available'):
                available_count +=1
                print("Workspace {} ({}) available".format(workspaces[i], i+1))
        
        return available_count == len(workspaces)


    def test_kfold(self, df_test, ws_id):
        """
        This function will take the regression test uploaded in csv and will send each phrase to WA and collect 
        information on how the system responded. 
        
        :param df_test: the dataframe containing the testing phrases 
        :param ws-id: the index of the fold that would be used to call the correct workspace id that needs to be test 
        :return results: a pandas dataframe with original text, predicted intent and also the results from WA
        """
        results = pd.DataFrame([],columns = ['original_text','expected intent','intent1',
                            'confidence1','intent2','confidence2','intent3',
                            'confidence3'])

        for i in tqdm(range(len(df_test))):

            text = df_test['utterance'][i]

            response = self.assistant.message(workspace_id=workspaces[ws_id], input={'text': text}, alternate_intents= True)
            dumps = json.dumps(response.get_result(), indent=2)
            
            data = json.loads(dumps)

            intent1= data['intents'][0]['intent']
            intent2= data['intents'][1]['intent']
            intent3= data['intents'][2]['intent']
            confidence1 = data['intents'][0]['confidence']
            confidence2 = data['intents'][1]['confidence']
            confidence3 = data['intents'][2]['confidence']

            results = results.append({
                    'original_text': df_test["utterance"][i],
                    'expected intent': df_test["intent"][i],
                    'intent1': intent1, 
                    'confidence1':confidence1, 
                    'intent2':intent2, 
                    'confidence2': confidence2, 
                    'intent3': intent3,
                    'confidence3': confidence3, 
                }, ignore_index=True)
            
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
        for i in tqdm(range(len(folds))):
            print("\n")
            print("RUNNING K-FOLD FOR FOLD NUMBER {}".format(i+1))
            test_index = folds[i]['test']
            df_test = self.training_df.iloc[test_index]
            df_test_reindexed = df_test.reset_index()
            results = self.test_kfold(df_test_reindexed, i)
            results["fold"] = i+1
            test_results = test_results.append(results)
        print("\n")
        print("FINISHED")

        test_results.loc[:, "intent_correct"] = test_results["intent1"]
        test_results["intent_correct"] = np.where((test_results["confidence1"]<threshold), "BELOW_THRESHOLD", test_results["intent1"])

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
        list_df["intent_correct"] = np.where((list_df["confidence1"]<threshold), "BELOW_THRESHOLD", list_df["intent1"])
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
        
        return report_df.T    

    def delete_kfold_workspaces(self, workspaces):
        """
        delete the workspaces when you dont need them anymore
        """
        for i in range(len(workspaces)):
            print("deleting workspace {} out of {}: {}".format(i+1, len(workspaces), workspaces[i]))
            response = self.assistant.delete_workspace(
                    workspace_id = workspaces[i]).get_result() 


if __name__ == "__main__":
    k_fold_number = 3 #USER INPUT 
    threshold = 0.4   #USER INPUT
    auth_type = 'apikey'
    apikey = '5XuD4BoYqV0Lx5PUltqzVnDue1MzCTgGcld8uCtPsNR_'
    url = 'https://gateway.watsonplatform.net/assistant/api'
    workspace_id = 'e6b17f68-9a81-4ac6-ae18-231717a47d3a'
    threshold = 0.4
    results_path = './kfold_results.csv' 

    kfold = kfoldtest(apikey=apikey, url=url, threshold=threshold, n_folds=k_fold_number)
    train_df = kfold.intent_df_from_watson(workspace_id)
    kfold.check_sufficient_workspaces()
    folds = kfold.create_folds()
    workspaces = kfold.create_kfold_WA(folds)

    available_flag = False
    
    while available_flag == False:
        print("Checking workspaces..")
        available_flag = kfold.check_status(workspaces)
        time.sleep(20)
    
    try:
        # results per utterance
        results_kfold = kfold.run_kfold_test(folds)
        results_kfold.to_csv('results_kfold.csv')

        # metrics per intent
        classification_report = kfold.create_classification_report(results_kfold)
        classification_report.to_csv('classification_report.csv')

        # metrics per fold
        metrics_per_fold = kfold.get_metrics_per_fold(results_kfold)
        metrics_per_fold.to_csv('metrics_per_fold.csv')

    finally:
        kfold.delete_kfold_workspaces(workspaces)
    
