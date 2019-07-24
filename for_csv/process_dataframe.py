"""
Collection of useful methods to preprocess a dataframe. 
- convert utterances to lowercase
- remove button clicks
- remove date/time utterances
- remove duplicate utterances

TODO:
- import list of buttons from central CSV
- get_df() function that checks whether the df is empty
"""
import pandas as pd

class process_dataframe(object):
    def __init__(self, df, utterance_col='utterance', conf1_col='confidence1_0'):
        self.df = df
        self.utterance_col = utterance_col
        self.utterance_lower_col = False
        self.conf1_col = conf1_col

    def utterance_col_to_lowercase(self):
        """
        Creates new col with _lower appended to its name, which is a 
        lowercase version of the utterance col.
        """
        self.utterance_lower_col = self.utterance_col + '_lower'

        df = self.df.copy()
        df[self.utterance_lower_col] = df[self.utterance_col].str.lower()

        self.df = df

        return df

    def string_remove_chars(self, string_in, chars_to_remove):
        """
        Remove characters in chars_to_remove from a string
        """
        for char in chars_to_remove:
            string_in = string_in.replace(char, '')    
        return string_in

    def string_is_date(self, string_in):
        """
        Returns whether a string can be parsed by the date parser
        """
        from dateutil.parser import parse
        try: 
            parse(string_in)
            return True
        except ValueError:
            return False
        except:
            print('Warning: ' + str(string_in) + ' could not be parsed by the date parser')
            return False

    def remove_numeric_utterances(self, chars_toignore=[]):
        """
        Drop utterances that are purely numeric.
        First drops any chars from each utterance that are 
        in the list chars_toignore.
        """
        if not self.utterance_lower_col:
            self.utterance_col_to_lowercase()

        # sort char list by longest first for easiest removal
        chars_toignore.sort(key = len, reverse=True) 

        idx_todrop = []

        for idx, utterance in self.df[self.utterance_lower_col].iteritems():
            if str(utterance) != 'nan':
                isdigit_stripped = self.string_remove_chars(utterance, chars_toignore)
                isdigit = isdigit_stripped.isnumeric()
                
                if isdigit:
                    idx_todrop.append(idx)

        self.df = self.df.drop(index=idx_todrop)

    def remove_date_utterances(self):
        """
        Drop utterances that are purely dates.
        """

        if not self.utterance_lower_col:
            self.utterance_col_to_lowercase()

        idx_todrop = []

        for idx, utterance in self.df[self.utterance_lower_col].iteritems():
            if str(utterance) != 'nan':
                isdate = self.string_is_date(utterance)
                
                if isdate:
                    idx_todrop.append(idx)

        self.df = self.df.drop(index=idx_todrop)

    def drop_utterances_containing(self, utterance_parts_to_remove, lower=True):
        """
        Drop utterances that are in a list, e.g. button clicks.
        By default this list should all be lowercase as it is 
        compared against lowercase utterances.
        """
        # TODO: make generic for columns

        if not self.utterance_lower_col:
            self.utterance_col_to_lowercase()
        
        if lower:
            self.df = self.df[~self.df[self.utterance_lower_col].str.contains('|'.join(utterance_parts_to_remove))]
        else:
            self.df = self.df[~self.df[self.utterance_col].str.contains('|'.join(utterance_parts_to_remove))]

    def get_utterances_containing_string(self, string_arg, lower=True):
        """
        Return rows with utterances containing a string argument.
        """
        if not self.utterance_lower_col:
            self.utterance_col_to_lowercase()

        if lower:
            return self.df[self.df[self.utterance_lower_col].str.contains(string_arg)]
        else:
            return self.df[self.df[self.utterance_col].str.contains(string_arg)]


    def drop_rows_with_column_value(self, column_name, toremove_list, lower=True, invert=False):
        """
        Drop rows which contain a value from a list in a certain column.
        By default this list is lowercase, and is compared against a 
            lowercase version of the column. 
        If the invert flag is used, this function will act as 
            keep_rows_with_column_value.
        """

        if not self.utterance_lower_col:
            self.utterance_col_to_lowercase()
        
        if lower:
            self.df = self.df[self.df[column_name].str.lower().isin(toremove_list) == invert]
        else:
            self.df = self.df[self.df[column_name].isin(toremove_list) == invert]

    def drop_duplicate_utterances(self, duplicate_thresh=1):
        """
        Drop utterances which are duplicated more than the number of times in specified by 
        duplicate_thresh.
        """
        if not self.utterance_lower_col:
            self.utterance_col_to_lowercase()
            
        gb_utterance = self.df.groupby(self.utterance_lower_col).count()
        idx_todrop_duplicates = gb_utterance[gb_utterance[self.utterance_col] > duplicate_thresh].index.tolist()

        self.df = self.df[~self.df[self.utterance_lower_col].isin(idx_todrop_duplicates)]

    def drop_confidence_greaterthan(self, confidence_thresh):
        """
        Drop utterances with a confidence greater than the confidence threshold.
        """
        self.df = self.df[self.df[self.conf1_col] <= confidence_thresh]

    def check_df_not_empty(self):
        """
        Raises a ValueError if len(df) == 0
        """
        if self.df.shape[0] == 0:
            raise ValueError('DataFrame empty (0 rows)')

    def get_df(self):
        """
        Checks whether the df is empty and returns it if it isn't.
        """

        try:
            self.check_df_not_empty()
            return self.df.copy()
        except:
            print('Dataframe empty, so nothing returned.')

# TODO: Write unit tests for functions
if __name__ == '__main__':
    import unittest

    data_dict = {
        "utterance": ["hello i am a string", "StRiNg WiTh UpPeRCaSe", "0892349482094", "14th feb 2018", "£24.59"]
    }

    df = pd.DataFrame.from_dict(data_dict)

    def testLowercase(df):
        process_df = process_dataframe(df, utterance_col='utterance')
        process_df.utterance_col_to_lowercase()

        data_dict_correct = {
        "utterance": ["hello i am a string", "StRiNg WiTh UpPeRCaSe", "0892349482094", "14th feb 2018", "£24.59"],
        "utterance_lower": ["hello i am a string", "string with uppercase", "0892349482094", "14th feb 2018", "£24.59"]
        }
        correct_df = pd.DataFrame.from_dict(data_dict_correct)

        print( pd.DataFrame.equals(correct_df, process_df.df) )

    testLowercase(df)