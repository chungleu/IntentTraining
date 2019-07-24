# *****************************************************************
#
# Licensed Materials - Property of IBM
#
# IBM Watson Conversation Test Suite
# (C) Copyright IBM Corp. 2017. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
#
# *****************************************************************

from watson_developer_cloud import ConversationV1
import pandas as pd
import itertools

class BlindTestV1:
    """Blind Testing"""

    def __init__(self, **kwargs):
        self.recall_levels = 10
        self.ctk = ConversationV1(**kwargs)

    def runBlindTest(self, workspace_id=None, questions=None,
                     get_data=False, show_progress=True,
                     limit_intents=None):
        """
        Runs a blind test with the questions specified.
        :param workspace_id The workspace to use.
        :param questions Pandas Dataframe of Question + Intent.
        :param get_data If set to true, then the report is sent back rather then results.
        :param show_progress If True will show a line for every check. If get_data is False, it will not show the question.
        :param limit_intents if given, retrieve only the corresponding amount of top intents for each test
        """
        
        full_results = pd.DataFrame(columns=[
            'Question','Expected Intent']+
                            list(itertools.chain(*[['Intent{}'.format(i),
                                                    'Confidence{}'.format(i),
                                                    'R@{}'.format(i)]
                                      for i in range(1, self.recall_levels+1)])))

        # test each question
        for index, row in questions.iterrows():
            question = row['Question']
            expected_intent = row['Intent']
            message = { 'text': question }
            
            response = self.ctk.message(workspace_id=workspace_id,
                                        message_input=message,
                                        alternate_intents=True)

            # allow to consider only top N intents (N=limit_intents)
            # this helps in performances when dealing with intent-rich workspaces
            if limit_intents:
                intents = response['intents'][:limit_intents]
            else:
                intents = response['intents']

            rec = { 'Question': question, 'Expected Intent': expected_intent }

            for i in range(0,len(intents)):
                rec.update({
                    'Intent{}'.format(i+1): intents[i]['intent'],
                    'Confidence{}'.format(i+1): intents[i]['confidence'],
                    'R@{}'.format(i+1): 0
                    })

            entities = response['entities']
            
            for i in range(0,len(entities)):
                rec.update({
                    'Entity{}'.format(i+1): entities[i]['entity'],
                    'EntityValue{}'.format(i+1): entities[i]['value'],
                    })
            
            for i in range(len(intents) -1, -1, -1):
                if rec['Expected Intent'] == rec['Intent{}'.format(i+1)]:
                    rec['R@{}'.format(i+1)] = 1
            
            if show_progress and get_data: 
                print('{} {}'.format(index+1, question))
            elif show_progress:
                print('{} <BLIND>'.format(index+1))

            full_results = full_results.append(rec,ignore_index=True)
        
        
        self.cache = full_results.fillna(0)
        if get_data:
            return self.cache
        
        results = pd.DataFrame(columns=['Total']+
                            list(itertools.chain(*[['R@{}'.format(i),
                                                    'R@{}%'.format(i)]
                                      for i in range(1, self.recall_levels+1)])))
        record = {'Total': full_results.shape[0]}
        
        for i in range(0,10):
            record.update({ 'R@{}'.format(i+1): full_results['R@{}'.format(i+1)].sum() })
        
        for i in range(0,10):
            r = record['R@{}'.format(i+1)]
            if r > 0: 
                record.update({ 'R@{}%'.format(i+1): r / record['Total'] })
            else:
                record.update({ 'R@{}%'.format(i+1): 0 })
        
        results = results.append(record,ignore_index=True)

        results = results.drop('Total',1)
        for i in range(0,10):
            results = results.drop('R@{}'.format(i+1), 1)

        return results
    
    def getFullResults(self, recall_per_row=False):
        """
        Returns the full results if they exist. Otherwise returns None.
        It is generated after runBlindTest is run.
        """
        try:
            self.cache
        except AttributeError:
            return None

        if recall_per_row == False: return self.cache
        
        results = pd.DataFrame(columns=['Question', 'Expected Intent', 'Recall@', 'Found', 'Confidence'])
        
        for index, row in self.cache.iterrows():  # @UnusedVariable
            record = {'Question': row['Question'], 'Expected Intent': row['Expected Intent'] }
            for i in range(0,10): 
                record.update({
                    'Recall@': i+1,
                    'Confidence': row['Confidence{}'.format(i+1)],
                    'Found': row['R@{}'.format(i+1)] == 1
                    })
                results = results.append(record,ignore_index=True)
                
        return results
        