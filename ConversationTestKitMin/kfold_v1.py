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
import time
import itertools


class KFoldTestV1:
    def __init__(self, **kwargs):
        self.recall_levels = 5
        self.max_folds = 20
        self.ctk = ConversationV1(**kwargs)

    def isWCSTraining(self, workspacelist):
        for ws in workspacelist:
            wsStatus = self.ctk.get_workspace(ws)

            if wsStatus['status'] != "Available":
                return True
        return False

    def deleteWorkspaces(self, workspacelist):
        for ws in workspacelist:
            self.ctk.delete_workspace(ws)

    def runFoldTest(self, questions=None, entities=None, kfoldNum=5, show_progress=True,
                    full_results=False, rerun_test=False, save_tests=False,
                    delete_workspaces=False, wcsLanguage="en"):
        """
        :param rerun_test:  True to rerun a test with the same test and training combinations
          Supplied question file must include a 'TestSetRef' column which references which test fold the question is in
        :param save_tests: True to save the test dataframe to a csv
         Contains the questions, intents and a 'TestSetRef' column which references which test fold the question is in
        :param full_results: Return dataframe with full test results
        :param wcsLanguage: Language parameter for the Conversations Workspaces
        :param questions: Pandas Dataframe of Question + Intent, + TestSetRef if rerun_test=True
        :param entities: list of entities to populate the Workspace with
        :param kfoldNum: Number of folds to use for the test, default is 10.
        :param show_progress: If True will show a line for every check.
        :param delete_workspaces: delete workspaces after testing is complete
        :return: Returns a results dataframe and the test dataframe if save_tests=True
        """
        try:
            if kfoldNum > self.max_folds:
                print("Number of folds cannot exceed {}".format(self.max_folds))
                raise SystemExit()

            if not rerun_test:
                questions = questions.sample(frac=1).reset_index(drop=True)
                questions['TestSetRef'] = questions.index % kfoldNum

            # create kfoldNum workspaces
            workspaceList = []
            for i in range(0, kfoldNum):
                testNode = [{"dialog_node": "kfoldtest", "description": "K-Fold Test Node",
                            "conditions": "context.kfoldTest == true", "output":
                                {"text": {"text": "K-Fold Test Node"}}}]
                intentList = self.build_intent_json(questions, i)
                if show_progress:
                    print('Creating Workspace {} of {} '.format(i + 1, kfoldNum))
                response = self.ctk.create_workspace(name="Test_" + str(i),
                                                    description="testws_" + str(i),
                                                    language=wcsLanguage,
                                                    dialog_nodes=testNode,
                                                    intents=intentList, entities=entities)
                workspaceList.append(response['workspace_id'])

            # wait until wcs finished training
            while self.isWCSTraining(workspaceList):
                print('Checking Workspaces status....  In Training')
                time.sleep(10)
            print('Workspaces ready. Running tests')

            # compute results
            combined_results = pd.DataFrame()
            for i in range(0, kfoldNum):
                testquestions = questions[questions['TestSetRef'] == i]
                kfresults = self.runTestSets(workspace_id=workspaceList[i],
                                            questions=testquestions, kfoldnum=i,
                                            show_progress=show_progress)
                combined_results = pd.concat([combined_results, kfresults], axis=0)
            testresults_recall = self.kFoldSummary(combined_results)

            if delete_workspaces == True:
                self.deleteWorkspaces(workspaceList)

            if save_tests and full_results:
                return combined_results, testresults_recall, questions
            elif full_results:
                return combined_results, testresults_recall
            elif save_tests:
                return  testresults_recall, questions
            else:
                return testresults_recall
        except Exception as e:
            self.deleteWorkspaces(workspaceList)
            print('Kfold tests failed to run, so the temporary workspaces have been deleted.')
            print(e)

    def runTestSets(self, workspace_id, questions, kfoldnum, show_progress=True):
        """
        Runs a blind test with the questions specified.
        :param workspace_id The workspace to use.
        :param questions Pandas Dataframe of Question + Intent.
        :param kfoldnum Sets number of folds in test.
        :param show_progress If True will show a line for every check.
        """
        context = {'kfoldTest': True, 'expected_intent': 'none'}
        testLength = len(questions.index)
        full_results = pd.DataFrame(
            columns=['KFold', 'Question', 'Expected Intent'] +
                        list(itertools.chain(*[['Intent{}'.format(i),
                                       'Confidence{}'.format(i),
                                       'Correct@{}'.format(i)]
                                      for i in range(1, self.recall_levels+1)])))

        # test each question
        for questionCount, (index, row) in enumerate(questions.iterrows()):
            if show_progress:
                print('Testing fold : {}, question {} of {}, question : {}'.format(
                                            kfoldnum+1, questionCount, testLength, question))

            question = row['Question']
            expected_intent = row['Intent']
            message = {'text': question}

            response = self.ctk.message(workspace_id=workspace_id,
                                        message_input=message,
                                        context=context, alternate_intents=True)

            intents = response['intents']
            rec = {'KFold': kfoldnum+1, 'Question': question, 'Expected Intent': expected_intent}
            for i in range(0, min(self.recall_levels, len(intents))):
                correct = intents[i]['intent'] == expected_intent
                rec.update({
                    'Intent{}'.format(i+1): intents[i]['intent'],
                    'Confidence{}'.format(i+1): intents[i]['confidence'],
                    'Correct@{}'.format(i+1): int(correct)
                })

            full_results = full_results.append(rec, ignore_index=True)

        # in case we have less intents than expected, drop corresponding empty columns
        return full_results.dropna(axis=1, how='all')

    def build_intent_json(self, questions, testsetnum):
        intentList = []
        questions = questions[questions['TestSetRef'] != testsetnum]
        intentDict = {k: g["Question"].tolist() for k, g in questions.groupby("Intent")}
        for key, value in intentDict.items():
            examplesList = []
            for item in value:
                examplesList.append({"text": item})
            intentList.append({"intent": key, "examples": examplesList})
        return intentList

    def kFoldSummary(self, results_df):
        recall_summary_df = pd.DataFrame(
            columns=['Fold Name', 'Total']+
                        list(itertools.chain(*[['R@{}'.format(i),
                                                'Correct@{}'.format(i)]
                                      for i in range(1, self.recall_levels+1)])))
        for fold, group in results_df.groupby('KFold'):
            # count number of correct predictions for each recall level
            correct = group.filter(regex="Correct@").sum()
            total = len(group)
            res = {'Fold Name':'Fold_{}'.format(int(fold)),
                   'Total':total}
            # compute recall percentage at each present level
            for i in range(1, len(correct)+1):
                # use string sintax cause correct[0:i].sum() might fail depending on order
                recall_sum = sum([correct['Correct@{}'.format(j)] for j in range(1, i+1)])
                res['Correct@{}'.format(i)] = correct['Correct@{}'.format(i)]
                res['R@{}'.format(i)] = recall_sum/total
            recall_summary_df = recall_summary_df.append(res, ignore_index=True)
        # in case we have less intents than expected, drop corresponding empty columns
        return recall_summary_df.dropna(axis=1, how='all')
