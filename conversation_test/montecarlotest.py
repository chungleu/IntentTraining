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
from conversation_test.kfoldtest import kfoldtest

@click.command()
@click.argument('topic', nargs=1)
@click.option('--no_folds', '-n', type=int, default=3, help="No. folds to run.")
@click.option('--sample_size', '-s', type=float, default=0.2, help="Proportion of training set to hold out as a test set.")
@click.option('--results_type', '-r', type=click.Choice(['raw', 'metrics_intent', 'all']), default='all', help='Whether to give raw results per utterance, metrics (per intent), or all available.')
@click.option('--conf_matrix', '-c', is_flag=True ,help='Whether to plot a confusion matrix.')
def run_mc(topic, no_folds, sample_size, results_type, conf_matrix):
    """
    Runs monte-carlo test using credentials in ../Credentials.py
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
    folds = kf.create_folds(method='monte-carlo', sample_size=sample_size)
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

if __name__ == "__main__":
    run_mc()
    
