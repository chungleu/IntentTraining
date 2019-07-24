from datasketch import MinHash, MinHashLSH


class MinHash_similarity(object):
    def __init__(self, threshold=0.4, num_perm=128, shingle_length=5):
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_length = shingle_length

    def import_dataframe(self, df_path):
        """
        Imports and cleans a text, label dataframe
        """

        import pandas as pd

        df = pd.read_csv(df_path, names=['Question', 'Intent'])
        df = df.dropna(axis=0)

        return df

    def similarity_threshold_bulk(self, df_library, df_query, only_positive=False, return_df=False):
        """
        Takes a dataframe of 'library' strings to query against, and a dataframe of query strings. 
        Gives these unique IDs.
        Transforms both the library and the query strings into minhash objects.
        If return_df==True then df_query will be returned with a column showing how many similar utterances 
            have been found in df_library.
        TODO: maybe use redis in production
        """
        from datasketch import MinHashLSH

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        data_library = self.dataframe_to_data_list(df_library, 'lib_')
        data_query = self.dataframe_to_data_list(df_query, 'query_')

        # use an insertion session to create an lsh object with all the lib data that can be queried
        with lsh.insertion_session() as session:
            for key, minhash in data_library:
                session.insert(key, minhash)

        # bulk query the data_query objects against lsh
        query_results = []
        df_query['no_similar'] = 0

        for key, minhash in data_query:
            query_result = lsh.query(minhash)
            query_result_length = len(query_result)

            if return_df:
                df_query.loc[key, 'no_similar'] = len(query_result)
            elif only_positive:
                # only need to care about only_positive if not returning a dataframe
                if query_result_length > 0:
                    query_results.append((key, query_result, query_result_length))
            else:
                query_results.append((key, query_result, query_result_length))

        if return_df:
            return df_query
        else:
            return query_results

    def dataframe_to_data_list(self, df, index_prefix, utterance_col='Question'):
        """
        Prefixes the index with index_prefix.
        Creates a new column in the dataframe with a minhash object for every string.
        Converts this to a list of tuples (index, m) that can easily be iterated through.
        """
        data_list = []
        df.index = str(index_prefix) + df.index.astype(str)

        for index, row in df.iterrows():
            m = self.minhash_from_string(row[utterance_col])
            temptuple = (index, m)
            data_list.append(temptuple)

        return data_list


    def minhash_from_string(self, input_string):
        """
        Generates minhash object from string
        """
        shingles = self.extract_shingles(input_string)
        shingle_set = set(shingles)

        m = MinHash(num_perm=self.num_perm)
        for i in shingle_set:
            m.update(i.encode('utf8'))

        return m

    def extract_shingles(self, document):
        """
        Returns list of shingles of length determined by the input.
        """
        shingles = [document[i:i+self.shingle_length] for i in range(len(document))][:-self.shingle_length]

        return shingles

if __name__ == '__main__':
    """
        Test class functionality
    """

    mh = MinHash_similarity(threshold=0.5, num_perm=512, shingle_length=5)
    df_lib = mh.import_dataframe('data/payments_questions.csv')
    df_query = mh.import_dataframe('data/payments_blindset.csv')

    """ data_lib = mh.dataframe_to_data_list(df_lib, 'lib_')
    data_query = mh.dataframe_to_data_list(df_lib, 'query_')
    """
    
    """     query_results = mh.similarity_threshold_bulk(df_lib, df_query, only_positive=True)
    
    def print_query_results(num, query_results, df_lib, df_query):
        for i in range(0, num):
            print('Query:')
            print(df_query.loc[query_results[i][0], 'Question'])
            print('Similar items in library:')
            print(df_lib.loc[query_results[i][1], 'Question'])
            print('')

    print_query_results(3, query_results, df_lib, df_query)
    """
    query_results = mh.similarity_threshold_bulk(df_lib, df_query, only_positive=True, return_df=True)
    print( query_results.sort_values('no_similar', ascending=False).head() )