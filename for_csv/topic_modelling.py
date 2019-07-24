"""
Modules for summarising a list of utterances, e.g. through topic modelling.
"""
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

class topic_model(object):
    def __init__(self, utterance_series, stopword_list=[]):
        self.utterance_series = utterance_series
        self.stopword_set = set(stopword_list)

    def display_topics(self, no_top_words):
        """
        Prints topics. Uses self.model, self.feature_names
        """

        for topic_idx, topic in enumerate(self.model.components_):
            print("Topic " + str(topic_idx + 1))
            print(" ".join([self.feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))

    def get_LDA_topics(self, no_topics, min_df=0.002, max_df=0.2, no_features=1000):
        """
        Returns LDA model and feature names
        """
        tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=no_features, stop_words=set(self.stopword_set))
        tf = tf_vectorizer.fit_transform(self.utterance_series)
        tf_feature_names = tf_vectorizer.get_feature_names()
        
        lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50, random_state=0).fit(tf)

        self.model = lda
        self.feature_names = tf_feature_names

        return lda, tf_feature_names

    def display_LDA_topics(self, no_topics, no_top_words, min_df=0.002, max_df=0.2, no_features=1000):
        """
        Displays top topics and their top words.
        """

        self.get_LDA_topics(no_topics, min_df=min_df, max_df=max_df, no_features=no_features)
        self.display_topics(no_top_words)





    