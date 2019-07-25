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

# internal
from config import data_dir
from Credentials import ctx, workspace_id, conversation_version, active_adoption
#import for_csv
#from logging import getLogger
#logger = getLogger('download_intents')

### CHANGE ME
workspaces_to_ignore = []
### 

@click.command()
@click.option('--proxy', '-p', is_flag=True, help='Currently disabled.')
def click_main(proxy):
    main(proxy)

def main(proxy):
    debug = False

    def workspace_df_to_csv(df, workspace_name):
        """Exports workspace dataframe to CSV"""
        file_name = workspace_name + '_questions.csv'
        script_dir = os.path.dirname(__file__)
        output_dir = os.path.join(script_dir, data_dir, 'workspace_training/')
    
        output_path = os.path.join(output_dir, file_name)
        df.to_csv(output_path, index=False, header=False)

    if proxy:
        print('Proxy requires a newer version of the Watson Assistant SDK than used in this module. Run this script on a machine'
        ' for which a proxy is not required.')
        pass
        
        """ from getpass import getpass
        import ssl

        local_username = input("file ID: ")
        local_password = getpass()

        http_proxy = "http://" + local_username + ":" + local_password + "@proxyarray.service.group:8080"
        https_proxy = "https://" + local_username + ":" + local_password + "@proxyarray.service.group:8080"
        http_config = { 
        "proxies": {"http"  : http_proxy, "https"  : https_proxy}
        }
        os.environ['http_proxy'] = http_proxy
        os.environ['https_proxy'] = https_proxy """

    from watson_developer_cloud import ConversationV1
    ctk = ConversationV1(url= ctx.get(active_adoption)['url'], username=ctx.get(active_adoption)['username'], password=ctx.get(active_adoption)['password'], version=conversation_version)
    
    workspace_info = ctk.list_workspaces()
    if debug:
        print({workspace["name"]: workspace["workspace_id"] for workspace in workspace_info["workspaces"] if workspace["workspace_id"] not in workspaces_to_ignore})

    for workspace in tqdm(workspace_info["workspaces"]):
        workspace_id = workspace["workspace_id"]

        if workspace_id in workspaces_to_ignore:
            continue

        workspace_name = workspace["name"]
        
        if debug:
            print(workspace_name)

        workspace = ctk.get_workspace(workspace_id, export=True)
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
    click_main()