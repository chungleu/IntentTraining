"""
Extracts the intents in all workspaces within an instance to CSV, to data/workspace_training.
Same format as using the download intents functionality within the Watson Assistant UI.
"""
import sys
sys.path.append('..')

# external
import os
import pandas as pd
import json
import click
from tqdm import tqdm
from watson_developer_cloud import AssistantV1

# internal
from config import data_dir
import Credentials
import for_csv
from logging import getLogger
logger = getLogger('download_intents')

### CHANGE ME
workspaces_to_ignore = []
### 

def main():
    def workspace_df_to_csv(df, workspace_name):
        """Exports workspace dataframe to CSV"""
        file_name = workspace_name + '_questions.csv'
        script_dir = os.path.dirname(__file__)
        output_dir = os.path.join(script_dir, data_dir, 'workspace_training/')
    
        output_path = os.path.join(output_dir, file_name)
        df.to_csv(output_path, index=False, header=False)

    active_adoption = Credentials.active_adoption
    instance_creds = Credentials.ctx[active_adoption]
    conversation_version = Credentials.conversation_version

    if 'apikey' in instance_creds:
        logger.debug("Authenticating (apikey)")
        ctk = AssistantV1(iam_apikey=instance_creds['apikey'], url=instance_creds['url'], version=conversation_version)

    elif 'password' in instance_creds:
        logger.debug("Authenticating (username/password)")
        ctk = AssistantV1(username=instance_creds['username'], password=instance_creds['password'], url=instance_creds['url'], version=conversation_version)
    
    workspace_info = ctk.list_workspaces().get_result()
    logger.debug({workspace["name"]: workspace["workspace_id"] for workspace in workspace_info["workspaces"] if workspace["workspace_id"] not in workspaces_to_ignore})

    for workspace in tqdm(workspace_info["workspaces"]):
        workspace_id = workspace["workspace_id"]

        if workspace_id in workspaces_to_ignore:
            continue

        workspace_name = workspace["name"]
    
        workspace = ctk.get_workspace(workspace_id, export=True).get_result()
        intents = workspace['intents']

        workspace_df = pd.DataFrame()

        for intent in intents:
            intent_name = intent['intent']
            utterances = [example['text'] for example in intent['examples']]

            intent_df = pd.DataFrame(utterances)
            intent_df = intent_df.rename(columns={0:'utterance'})

            intent_df['Intent'] = intent_name

            workspace_df = workspace_df.append(intent_df)

        workspace_df_to_csv(workspace_df, workspace_name)

if __name__ == "__main__":
    main()