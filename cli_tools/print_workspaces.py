# prints a list of the available workspaces for the connected instance.
import os
import pandas as pd
import json
from tqdm import tqdm
from ibm_watson import AssistantV1
import sys

sys.path.append('..')
# internal
import Credentials
from logging import getLogger
logger = getLogger('print_workspaces')

def main():
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

    for item in workspace_info['workspaces']:
        print(item['name'], item['workspace_id'])

if __name__ == "__main__":
    main()