# *****************************************************************
#
# Licensed Materials - Property of IBM
#
# IBM Watson Conversation Test Suite
# (C) Copyright IBM Corp. 2017. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
#
# *****************************************************************

import pandas as pd
from watson_developer_cloud import ConversationV1
import json
import datetime

class WorkspaceV1:
    """ Class for Conversation Workspace"""
    CONV_SERVICE_VERSION = '2017-02-03'

    def __init__(self, workspace=None):
        self.conv = workspace

    @classmethod
    def get_from_local(cls, filepath):
        """
        Load workspace from local json file
        :param filepath:
        :return: workspace class instance
        """
        workspace = cls(WorkspaceV1.load_workspace(filepath))
        return workspace

    @classmethod
    def get_from_remote(cls, username, password, workspace_id):
        """
        Load workspace hosted on Bluemix. Workspace file will retrieve and then stored locally
        :param username: conversation service username
        :param password: conversation service password
        :param workspace_id: workspace to load
        :return: workspace class instance
        """
        conv_service = ConversationV1(
            username=username,
            password=password,
            version=WorkspaceV1.CONV_SERVICE_VERSION
        )
        workspace = cls(WorkspaceV1.download_workspace(conv_service, workspace_id))
        return workspace

    def get_intents(self):
        """
        Get the intents data (ground truth) for this conversation
        :return: a dataframe with one entry for each intent-question pair
        """
        intents = self.conv['intents']

        gt = []
        for intent in intents:
            for q in intent['examples']:
               gt.append((intent['intent'], q['text']))

        return pd.DataFrame(gt, columns=['intent', 'question'])

    def get_entities(self):
        """
        Get the entities data for this conversation
        :return: a dataframe that provides for each entity and for each entity
        value the corresponding list of synonyms
        """
        entities = self.conv['entities']

        data = []
        for entity in entities:
            entity_name = entity['entity']
            for val in entity['values']:
                data.append((entity_name, val['value'], val['synonyms']))

        return pd.DataFrame(data, columns=['entity', 'value', 'synonyms'])

    def get_nodes(self):
        """
        Get basic nodes info as flat dataframe
        :return: datafrane with basic info for each node (e.g. id, conditions, parent, prev sibling)
        """
        nodes = self.conv['dialog_nodes']

        data = []
        for node in nodes:
            data.append((node['dialog_node'], node['conditions'],
                         node['parent'], node['previous_sibling']))

        return pd.DataFrame(data, columns=['node_id', 'conditions', 'parent', 'previous_sibling'])

    def duplicate_to(self, username, password, workspace_name, description="", language='english'):
        """
        Create a copy of the current workspace and upload it using the provided credentials and info
        :param username: conversation service username
        :param password: conversation service password
        :param workspace_name: name of the new workspace
        :param description: description for the new workspace (default to empty string)
        :param language: language of the new workspace (default to english)
        :return: the id of the newly created workspace
        """
        conv_service = ConversationV1(
            username=username,
            password=password,
            version=WorkspaceV1.CONV_SERVICE_VERSION
        )
        response = conv_service.create_workspace(name=workspace_name, description=description,
                                             language=language, dialog_nodes=self.conv['dialog_nodes'],
                                             intents=self.conv['intents'])
        return response['workspace_id']

    def insert_node(self, parentId, nodeId, text="", conditions="", previousSiblingId=None):
        """
        Insert a new node into current loaded conversation
        :param parentId:
        :param nodeId:
        :param text:
        :param conditions:
        :param previousSiblingId:
        :return:
        """
        self.conv['dialog_nodes'].append(WorkspaceV1.generate_node(nodeId, text, conditions,
                                                                   previousSiblingId, parentId))

    def save_workspace_tofile(self, outpath):
        with open(outpath, 'w+', encoding='utf8') as f:
            f.write(json.dumps(self.conv))

    @staticmethod
    def generate_node(nodeId, text, conditions, previousSibling=None, parent=None):
        output = {'text': text}
        node = {
            'go_to': None,
            'output': output,
            'parent': parent,
            'created': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'previous_sibling': previousSibling,
            'description': None,
            'dialog_node': nodeId,
            'conditions': conditions,
            'context': None,
            'metadata': None

        }
        return node

    @staticmethod
    def download_workspace(conv_service, workspace_id):
        workspace = conv_service.get_workspace(workspace_id, export=True)
        return workspace

    @staticmethod
    def load_workspace(filepath):
        with open(filepath, 'r', encoding='utf8') as f:
            workspace = json.load(f)
        return workspace