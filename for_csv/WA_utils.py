"""
Collection of utils for working with Watson Assistant in Python
"""
import json
import pandas as pd
import time

import sys
sys.path.append('..')

import for_csv.logger
from logging import getLogger
logger = getLogger("WA_utils")


def create_intents(assistant, train_df):
    """
    It collects the intents in json format to send when creating the workspace 
        
    :param train_index: that are the results of the 'create_folds' function
    :return intent_results: if a list of dictionaries that will be sent when new workspace will be created
    """
    
    intent_results = []
    for i, _ in train_df.iterrows():
        row = {}
        text = train_df.iloc[i]['utterance']
        intent = train_df.iloc[i]['intent']

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

def create_workspace(assistant, name, intents_json, description=""):
    """
    create one skill 
    
    :param intent_json : output of the 'create_intents' function
    :param name : the name of the workspace to be created
    :return workspace_id: the id of the skill that has been generated
    """
    response = assistant.create_workspace(
        name=name,
        #language = 'en'   # CHANGE LANGUAGE HERE (Default is 'en')
        description=description, 
        intents = intents_json
    ).get_result()
    
    workspace_id = response.get('workspace_id')
    
    return workspace_id

def create_workspace_from_df(assistant, name, train_df, description="", poll_interval=20):
    """
    Wraps create_intents and create_workspace_from_df, then returns the skill ID when it has been trained.
    """

    intent_json = create_intents(assistant, train_df)
    skill_id = create_workspace(assistant, name, intent_json, description)

    status = ""

    while status != 'Available':
        logger.info("Waiting for {} skill to finish training..".format(name))
        status = assistant.get_workspace(workspace_id = skill_id).get_result()['status']
        time.sleep(poll_interval)

    return skill_id

def delete_workspace(assistant, workspace_id):
    """
    Deletes a workspace given the skill ID.
    """

    response = assistant.delete_workspace(workspace_id = workspace_id).get_result()

    return response
    # TODO: return that deletion has been successful (status code)

def get_training_data(assistant, workspace_id):
    """
    Returns training data for a skill as dataframe with columns utterance, intent 
    """

    response = assistant.list_intents(workspace_id=workspace_id)
    obj = json.dumps(response.get_result(), indent=2)
    data = json.loads(obj)
    
    df = pd.DataFrame(columns = ['utterance','intent'])
    
    for i in range(len(data["intents"])): 
        name_intent = data["intents"][i]["intent"]

        # Call WA to get the list of Examples of each intent 
        response = assistant.list_examples(workspace_id = workspace_id, intent = name_intent)
        dumps = json.dumps(response.get_result(), indent=2)
        data_examples = json.loads(dumps)

        # get the Ground Truth (examples test) of each intent 
        for j in range(len(data_examples["examples"])): 
            text = data_examples["examples"][j]["text"]
            df = df.append({'intent':name_intent,'utterance': text},ignore_index=True)
    
    return df