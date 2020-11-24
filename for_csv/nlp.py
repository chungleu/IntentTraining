"""
Various utils for NLP
"""

import nltk
from nltk.corpus import stopwords
import re
import pandas as pd

import fuzzywuzzy.process as fuzz_process
from fuzzywuzzy import fuzz


def extract_ngrams(sentence, n, stopwords_list='_none', chars_remove=None):
    """
    Returns a list of ngrams in a sentence, after removing a list of 
    stopwords.
    Args:
        - sentence (str): the string to extract ngrams from
        - n (int): the number of words in each ngram to extract
        - stopwords_list (list): any stopwords to not consider as ngrams. 
        - chars_remove (str): characters to drop from the string before extracting ngrams

    stopwords_list behaviour: 
        - None: use default nltk stopwords
        - list: use custom list, plus default nltk stopwords
        - '_none': override default behaviour to use no stopwords at all
    """

    if stopwords_list == '_none':
        stopword_set = {}
    else:
        stopword_set = set(stopwords.words('english'))

        if stopwords_list:
            stopword_set.update(set(stopwords_list))

    sentence = sentence.lower()
    chars_remove = '[' + chars_remove + ']'
    sentence = re.sub(chars_remove, '', sentence)
    tokens = [token for token in sentence.split(
        ' ') if token != '' and token not in stopword_set]

    all_ngrams = nltk.ngrams(tokens, n)
    ngram_list = []

    for ngram in all_ngrams:
        if any(token not in stopword_set for token in ngram):
            ngram_list.append(' '.join(ngram))

    return ngram_list


class ngrams_df(object):
    """
    One instance per df
    """

    def __init__(self, df, stopwords_in=[], chars_remove="", utterance_col='utterance'):
        self.df = df
        self.stopwords = stopwords_in
        self.chars_remove = chars_remove
        self.utterance_col = utterance_col

    def create_ngram_cols(self, n_list, separate_cols=True):
        """
        Creates ngram_i columns for each n in n_list, containing
        a list of ngrams within the string.
        """

        try:
            self.n_set.update(set(n_list))
        except:
            self.n_set = set(n_list)

        tempdf = self.df.copy()
        col_names = ['ngram_' + str(n) for n in n_list]

        for idx, n in enumerate(n_list):
            tempdf[col_names[idx]] = tempdf[self.utterance_col].apply(
                extract_ngrams, n=n, stopwords_list=self.stopwords, chars_remove=self.chars_remove)

        tempdf['ngrams_all'] = tempdf[col_names].sum(axis=1)

        self.df = tempdf
        return self.df

    def list_from_ngram_cols(self, n_list=False):
        """
        Returns a set of all ngrams from df cols containing lists of ngrams.
        Run after create_ngram_cols.
        """

        ngram_list = []

        if not n_list:
            n_set = self.n_set
        else:
            n_set = set(n_list)

        for n in n_set:
            col_name = 'ngram_' + str(n)
            column_list = [ngram for ngram_list in self.df[col_name]
                           for ngram in ngram_list]
            ngram_list.extend(column_list)

        return ngram_list

    def get_ngram_list(self, n_list):
        """
        Combines the above two functions to get a set of ngrams straight from
        a dataframe containing an utterance column.
        """

        self.create_ngram_cols(n_list)
        ngram_list = self.list_from_ngram_cols()

        return ngram_list

    def get_ngram_frequencies(self, n_list, top_a=False, norm=False, norm_thresh=False):
        """
        Returns a dataframe with ngrams, their value of n, and their absolute or normalised frequency.
        Params:
            - top_a: returns top a ngrams for each value of n by frequency. Inactive if norm_thresh is specified.
            - norm: normalises ngram frequencies per value of n by dividing by the max
            - norm_thresh: sets a minimum value of norm_count. Only active if norm is True

        # TODO: take log of frequencies?
        """

        if norm_thresh:
            norm = True

        freq_df = pd.DataFrame()
        self.create_ngram_cols(n_list)

        for n in n_list:
            temp_ngram_list = self.list_from_ngram_cols(n_list=[n])

            if norm_thresh or not top_a:
                # top_a not used if norm_thresh is specified
                temp_df = pd.Series(temp_ngram_list).value_counts()
            else:
                temp_df = pd.Series(temp_ngram_list).value_counts()[0:top_a]

            if norm:
                temp_df = temp_df / temp_df.max()
                temp_df = temp_df.rename_axis(
                    'ngram').reset_index().rename(columns={0: 'count_norm'})
                if norm_thresh:
                    temp_df = temp_df[temp_df['count_norm'] >= norm_thresh]
            else:
                temp_df = temp_df.rename_axis(
                    'ngram').reset_index().rename(columns={0: 'count'})

            temp_df['n'] = n

            freq_df = freq_df.append(temp_df)

        return freq_df


def fuzzy_match_lists(input_str, match_list, score_thresh=90, return_name=False):
    """
    Checks whether input_str is similar to any item in match_list.
    score_t = scoring function
    return_name: return the closest match if there is a match. By default the output is just True/False
    90 for fuzz.ratio considered almost an exact match
    Example of usage:
    master_df['response_match'] = master_df['response'].apply(self.fuzzy_match_lists, match_list=failed_responses_list, score_t=fuzz.ratio)
    """
    input_str = str(input_str)
    new_name, score, idx = fuzz_process.extractOne(
        input_str, match_list, scorer=fuzz.ratio)
    if score > score_thresh:
        if return_name:
            return new_name
        else:
            return True
    else:
        return False


if __name__ == '__main__':
    test_string = "just went to india - i had a great time"

    import sys
    sys.path.append('..')
    from config import chars_remove

    ngrams = extract_ngrams(test_string, 3, chars_remove=chars_remove)

    print(ngrams)
