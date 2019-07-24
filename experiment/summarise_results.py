import os
import importlib
import pandas as pd

class summarise_results(object):
    def __init__(self, workspace_conf_thresh, method, topic, data_folder='./results/'):
        self.workspace_conf_thresh = workspace_conf_thresh
        self.data_folder = data_folder
        self.method = method
        self.topic = topic

    def join_results_individual_method(self, export_csv=False):
        """
        Get all results from a named method and compile them all into one dataframe, with
        the 'iteration' column showing the iteration number.
        Optional csv export
        """

        fs = [f for f in os.listdir(self.data_folder) if f.startswith(self.method)]

        df = pd.DataFrame()
        for filename in fs:
            file_path = self.data_folder + filename

            tempdf = pd.read_csv(file_path)
            
            iteration = filename.split("_", -1)[1][2]
            tempdf['iteration'] = iteration
            
            df = pd.concat([df, tempdf])

        if export_csv:
            df.to_csv(self.data_folder + 'join_' + self.topic + '_' + self.method + '.csv')

        return df

    def calculate_confusion(self, results_df):
        """
        Calculates a confusion column for a results dataframe, with one TP, FP, FN, FN in
        the 'Confusion' column for each row 
        """

        results_df['Confusion'] = 'none'

        results_df.loc[((results_df['R@1'] == 1) & (results_df['Confidence1'] > self.workspace_conf_thresh)), 'Confusion'] = 'TP'
        results_df.loc[((results_df['R@1'] == 0) & (results_df['Confidence1'] > self.workspace_conf_thresh)), 'Confusion'] = 'FP'
        results_df.loc[((results_df['R@1'] == 1) & (results_df['Confidence1'] < self.workspace_conf_thresh)), 'Confusion'] = 'FN'
        results_df.loc[((results_df['R@1'] == 0) & (results_df['Confidence1'] < self.workspace_conf_thresh)), 'Confusion'] = 'TN'

        return results_df

    def calculate_stats_per_intent(self, results_df):
        """
        Given a results dataframe (for one iteration), outputs a table with columns
        [TP, FP, TN, FN, size, accuracy, precision, recall, F1]
        """
        df = results_df.copy()
        df = self.calculate_confusion(df)

        confmat = df.groupby(['Expected Intent', 'Confusion']).count()['Question'].unstack().fillna(0)

        confmat_required_cols = ['TP', 'FP', 'TN', 'FN', 'size', 'accuracy', 'precision', 'recall', 'F1']
        confmat_missing_cols = set(confmat_required_cols) - set(confmat.columns.values)

        for new_col in confmat_missing_cols:
            confmat[new_col] = 0

        cols = ['size', 'accuracy', 'precision', 'recall', 'F1']
        for col in cols:
            confmat[col] = 0
        
        intents = df['Expected Intent'].unique().tolist()
        intent_dist = df.groupby('Expected Intent').count()['Question']

        for intent in intents:
            confmat.loc[intent,'size'] = intent_dist[intent]
            confmat.loc[intent,'accuracy'] = (confmat.loc[intent,'TP'] + confmat.loc[intent,'TN']) / intent_dist[intent]
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

        return confmat[['size', 'accuracy', 'precision', 'recall']]

    def calculate_stats_per_intent_all_iterations(self, results_df, export_csv=False):
        iterations = results_df['iteration'].unique().tolist()

        confmat = pd.DataFrame()

        for iteration in iterations:
            temp_df = results_df[results_df['iteration'] == iteration]
            temp_confmat = self.calculate_stats_per_intent(temp_df)
            temp_confmat['iteration'] = iteration

            confmat = pd.concat([confmat, temp_confmat])

        if export_csv:
            confmat.to_csv(self.data_folder + 'confmat_' + self.topic + '_' + self.method + '.csv')

        return confmat


if __name__ == "__main__":
    method = 'lowconf'
    topic = 'test'
    summarise = summarise_results(workspace_conf_thresh=0.4, method=method, topic=topic)
    df = summarise.join_results_individual_method(export_csv=True)
    
    # confmat for one iteration
    """ df_it1 = df[df['iteration'] == '1']
    confmat = summarise.calculate_stats_per_intent(df_it1)
    print(confmat) """

    # confmat for all iterations
    confmat_allit = summarise.calculate_stats_per_intent_all_iterations(df)
    print(confmat_allit)