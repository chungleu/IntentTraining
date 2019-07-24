import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn import preprocessing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import itertools

class ConfusionMatrix:
    def __init__(self, workspace_thresh=False):
        self.workspace_thresh = workspace_thresh

    def create(self, results, fig_path='confusion_matrix.png'):
        """
        Main function to produce the confusion matrix and save it to a file
        """
        df_ord = self.create_ordinal_frame(results)

        d_test = df_ord['Expected Intent']
        d_pred = df_ord['Intent1']

        conf_mat = confusion_matrix(d_test, d_pred)

        np.set_printoptions(precision=2)

        fig = plt.figure(figsize=(12,12))
        self.plot(conf_mat, classes=self.all_intents, title='Confusion matrix')

        fig.savefig(fig_path)
       
    def create_ordinal_frame(self, results):
        # create convert labels to ordinals so sklearn conf matrix can use them
        df = results[['Expected Intent', 'Intent1', 'Confidence1']].copy()

        if self.workspace_thresh:
            df.loc[(df['Confidence1'] < self.workspace_thresh) & (df['Expected Intent'] == df['Intent1']), 'Intent1'] = 'z_lowconf'

        self.all_intents = sorted(list(set(df['Expected Intent'].unique().tolist()).union(set(df['Intent1'].unique().tolist()))))

        le = preprocessing.LabelEncoder()
        le.fit(self.all_intents)

        df_ord = pd.DataFrame(columns=df.columns.values)
        df_ord['Expected Intent'] = le.transform(df['Expected Intent'])
        df_ord['Intent1'] = le.transform(df['Intent1'])
        df_ord['Confidence1'] = df['Confidence1'].copy()

        return df_ord

    def plot(self, cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        #plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
