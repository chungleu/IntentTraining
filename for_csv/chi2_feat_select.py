"""
To get most representative ngrams using chi2 feature selection
"""

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

class chi2_feat_select(object):
    """
    Two functions: 
    - do_tfidf performs tfidf for any ngram range
    - getKbest runs do_tfidf for unigrams & bigrams and returns a df with the most distinguishing of each for each intent
    """
    def __init__(self, train_df, utterance_col='utterance', intent_col='intent'):
        self.train_df = train_df
        self.utterance_col = utterance_col
        self.intent_col = intent_col

    def do_tfidf(self, ngram_range=(1,2)):
        """
        Returns feature matrix and labels from text.
        Doesn't change self.train_df
        """
        train_df = self.train_df

        id_col_name = self.intent_col + '_id'
        train_df[id_col_name] = train_df[self.intent_col].factorize()[0]
        self.intent_id_df = train_df[[self.intent_col, id_col_name]].drop_duplicates().sort_values(id_col_name)
        self.intent_to_id = dict(self.intent_id_df.values)
        self.id_to_intent = dict(self.intent_id_df[[id_col_name, self.intent_col]].values)


        self.tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', ngram_range=ngram_range, stop_words='english')
        # Sublinear_tf so that twenty times the occurrences of a word doesn't carry twenty times the weight. 
        
        features = self.tfidf.fit_transform(train_df[self.utterance_col]).toarray()
        labels = train_df[id_col_name]
        
        return features, labels

    def getKbest(self, K=5):
        """
        Takes train_df as input and returns df with K best ngrams for each class
        """

        features, labels = self.do_tfidf()

        chi2_dict = dict()

        for intent, intent_id in sorted(self.intent_to_id.items()):
            features_chi2 = chi2(features, labels==intent_id)
            indices = np.argsort(features_chi2[0])
            feat_names = np.array(self.tfidf.get_feature_names())[indices]

            unigrams = [f for f in feat_names if len(f.split(' ')) == 1]
            bigrams = [f for f in feat_names if len(f.split(' ')) == 2]

            chi2_dict[intent] = dict()
            chi2_dict[intent]['top unigrams'] = str(unigrams[-K:]).strip("[]").replace("'", "")
            chi2_dict[intent]['top bigrams'] = str(bigrams[-K:]).strip("[]").replace("'", "")
            
        return pd.DataFrame.from_dict(chi2_dict).reindex(['top unigrams', 'top bigrams'])
            