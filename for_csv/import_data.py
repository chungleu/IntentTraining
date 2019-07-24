"""
Various functions to import datasets. 
"""
# TODO: move all modules to use these functions

# external
import pandas as pd
import os

# internal
from config import *
from for_csv import utils

def get_train_test_data(topic, data_type):
    """
    Get and clean training/test data for a specified topic. 
    If topic==None for training data, then a dataframe with 
    all the topics specified in config is returned, with a 
    'topic' column indicating the topic. 
    """

    if data_type == 'test':
        file_ext = '_blindset.csv'
    elif data_type == 'train':
        file_ext = '_questions.csv'

    if (topic == None) & (data_type != 'test'):
        df = join_all_training_data()
    else:
        from for_csv import utils
        utils = utils(topic)

        file_name = topic + file_ext
        training_path = os.path.join(training_dir, file_name)
        df = utils.import_training_data(training_path)

    return df

def join_all_training_data():
    """
    Get training data for all topics and return a dataframe with a column 'topic',
    which specifies which workspace the training comes from.
    """

    df_return = pd.DataFrame()

    for topic in workspace_list:
        topic = 'divr-' + topic
        tempdf = get_train_test_data(topic, 'train')
        tempdf['topic'] = topic

        df_return = df_return.append(tempdf)

    return df_return

def get_test_data(topic):
    """
    Get and clean test data for a specified topic, for out 
    of scope utterance detection. If topic==None then a dataframe 
    with all the topics specified in config is returned, with a 
    'topic' column indicating the topic. 
    """

    from for_csv import utils
    utils = utils.utils(topic)

    file_name = topic + '_blindset.csv'
    training_path = os.path.join(training_dir, file_name)
    df_training = utils.import_training_data(training_path)

    return df_training
