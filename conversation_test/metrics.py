"""
Calculates TP, FP etc as well as precision, recall and accuracy given a confidence threshold.
"""

import pandas as pd
import time
import numpy as np
# hide warnings for divide by zero
np.seterr(divide='ignore', invalid='ignore')

def move_index_to_top(df, index_to_move):
    """
    Moves a row with a certain index to the top of a dataframe.
    Used in Metrics.get_all_metrics_CV() so the workspace value is at the top.
    """

    index_vals = df.index.values.tolist()
    index_vals.remove(index_to_move)
    index_vals = sorted(index_vals)

    return df.reindex([index_to_move, *index_vals])

class Metrics(object):
    def __init__(self, workspace_thresh, rat1_col="R@1"):
        """
        workspace_thresh (num < 1): confidence threshold for workspace
        """
        self.workspace_thresh = workspace_thresh
        self.rat1_col = rat1_col

    def calculate_workspace_metrics(self, results):
        """
        Returns set size, precision, recall, accuracy and F1 for workspace overall. 
        Also returns results df with column specifying TP/FP/TN/FN.
        """
        df = results.copy()
        df['confusion'] = 'none'
        df[self.rat1_col] = df['expected intent'] == df['intent1']
        df.loc[((df[self.rat1_col] == True) & (df['confidence1'] > self.workspace_thresh)), 'confusion'] = 'TP'
        df.loc[((df[self.rat1_col] == False) & (df['confidence1'] > self.workspace_thresh)), 'confusion'] = 'FP'
        df.loc[((df[self.rat1_col] == True) & (df['confidence1'] < self.workspace_thresh)), 'confusion'] = 'FN'
        df.loc[((df[self.rat1_col] == False) & (df['confidence1'] < self.workspace_thresh)), 'confusion'] = 'TN'

        confmat = df.groupby('confusion').count()['original_text']

        for item in ['TP', 'FP', 'FN', 'TN']:
            if item not in confmat.index:
                confmat[item] = 0

        # calculate accuracy, precision, recall, F1
        performance_stats = dict()
        performance_stats['threshold'] = self.workspace_thresh
        performance_stats['set size'] = len(df)
        performance_stats['accuracy'] = (confmat['TP'] + confmat['TN']) / len(df)
        performance_stats['precision'] = confmat['TP'] / (confmat['TP'] + confmat['FP'])
        performance_stats['recall'] = confmat['TP'] / (confmat['TP'] + confmat['FN'])
        performance_stats['F1'] = 2*(performance_stats['precision'] * performance_stats['recall'])/(performance_stats['precision'] + performance_stats['recall'] )  

        performance_stats_df = pd.DataFrame.from_dict(performance_stats, orient='index').T 
        self.workspace_index = 'workspace average (weighted)' 
        performance_stats_df.index = [self.workspace_index]

        results_with_confusion = df

        return performance_stats_df, results_with_confusion

    def calculate_metrics_per_intent(self, results, detailed_results=False, average_over_intents=True):
        """
        Returns set size, precision, recall, accuracy and F1 for each intent.
        detailed_results: enables/disables TP/FP/TN/FN in output
        """

        df = results.copy()
        
        # calculate stats per workspace if not done already
        if 'confusion' not in df.columns.values:
            _, df = self.calculate_workspace_metrics(df)
        
        expected_intent_dist = df.groupby('expected intent').count()['original_text']
        intent1_dist = df.groupby('intent1').count()['original_text']
        intents = df['intent1'].unique().tolist()
        
        # calculate accuracy, precision, recall, F1 for each intent
        confmat = df.groupby(['expected intent', 'confusion']).count()['original_text'].unstack().fillna(0)
        for item in ['TP', 'FP', 'FN', 'TN']:
            if item not in confmat.columns:
                confmat[item] = 0
                
        cols = ['set size', 'accuracy', 'precision', 'recall', 'F1']
        for col in cols:
            confmat[col] = 0
            
        for intent in intents:
            try:
                confmat.loc[intent,'set size'] = expected_intent_dist[intent]
            except:
                print("Intent {} not found in test set".format(intent))
                confmat.loc[intent,'set size'] = 0 # intent has been found that didn't exist in test set

            try:
                confmat.loc[intent,'accuracy'] = (confmat.loc[intent,'TP'] + confmat.loc[intent,'TN']) / expected_intent_dist[intent]
            except:
                confmat.loc[intent,'accuracy'] = 'nan'

            try:
                confmat.loc[intent,'precision'] = confmat.loc[intent,'TP'] / (confmat.loc[intent,'TP'] + confmat.loc[intent,'FP'])
            except:
                confmat.loc[intent,'precision'] = 'nan'
                
            try:
                confmat.loc[intent,'recall'] = confmat.loc[intent,'TP'] / (confmat.loc[intent,'TP'] + confmat.loc[intent,'FN'])
            except:
                confmat.loc[intent,'recall'] = 'nan'

            try:    
                confmat.loc[intent,'F1'] = 2*(confmat.loc[intent,'precision'] * confmat.loc[intent,'recall'])/(confmat.loc[intent,'precision'] + confmat.loc[intent,'recall'])
            except:
                confmat.loc[intent,'F1'] = 'nan'
            
        performance_stats_intent_df = confmat
        performance_stats_intent_df['threshold'] = self.workspace_thresh

        if not detailed_results:
            performance_stats_intent_df = performance_stats_intent_df[['threshold', 'set size','accuracy', 'precision', 'recall', 'F1']]

        if average_over_intents:
            cols_for_average = cols + ['threshold']
            performance_stats_intent_df.loc['workspace average (intents)', cols_for_average] = performance_stats_intent_df.loc[:, cols_for_average].mean(axis=0)
            performance_stats_intent_df = move_index_to_top(performance_stats_intent_df, 'workspace average (intents)')

        return performance_stats_intent_df

    def get_all_metrics(self, results, detailed_results=False):
        """
        Returns a dataframe with both the workspace-level metrics and metrics for each intent.
        """

        stats_workspace, results_with_confusion = self.calculate_workspace_metrics(results)
        stats_intent = self.calculate_metrics_per_intent(results_with_confusion, detailed_results)

        stats_combined = pd.concat([stats_workspace, stats_intent], sort=False)

        return stats_combined, results_with_confusion

    def get_all_metrics_CV(self, results, fold_col, detailed_results=False):
        """
        Variant of get_all_metrics for cross-validation tests which calculates the average and 
        range for each metric over folds.
        Doesn't return results_with_confusion as confusion takes a different meaning for each fold.
        Returns a dataframe with both the workspace-level metrics and metrics for each intent.
        """
        folds = results[fold_col].unique().tolist()
        
        stats_combined = pd.DataFrame()

        for fold in folds:
            stats_workspace, results_with_confusion = self.calculate_workspace_metrics(results)
            stats_intent = self.calculate_metrics_per_intent(results_with_confusion, detailed_results)

            stats_combined_temp = pd.concat([stats_workspace, stats_intent], sort=False)
            stats_combined_temp[fold_col] = fold

            stats_combined = stats_combined.append(stats_combined_temp)

        stats_combined = stats_combined.groupby(stats_combined.index).mean().drop(columns=fold_col).pipe(move_index_to_top, index_to_move=self.workspace_index)

        return stats_combined

if __name__ == "__main__":

    # UNIT TEST FUNCTIONS
    def test_get_all_metrics_CV(input_path, detailed_results):
        print('Testing function test_get_all_metrics_CV')
        results = pd.read_csv(input_path)
        metrics = Metrics(workspace_thresh=0.4)
        stats_combined_out = metrics.get_all_metrics_CV(results, fold_col='KFold', detailed_results=detailed_results)

        print(stats_combined_out)

    #####

    # RUN TESTS
    mc_results_path = "../run_tests/output/travel_mc_results_20190524-1358.csv"
    test_get_all_metrics_CV(mc_results_path, detailed_results=False)