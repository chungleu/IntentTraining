"""
Utils for explainability of the text classifier. 
"""

from lime.lime_text import LimeTextExplainer
import json
import numpy as np

class lime_explainer(object):
    def __init__(self, assistant, workspace_id):
        self.assistant = assistant # authenticated WA object
        self.workspace_id = workspace_id

    def classify_text(self, input_text: str, return_classes=False):
        """
        return_classes(bool): whether classes are returned
        """

        if not isinstance(input_text, (list,)):
            # classify string
            classifieroutput = self.assistant.message(
                workspace_id=self.workspace_id,
                input={
                    'text': input_text
                },
                alternate_intents=True
            ).get_result()

            # convert output so usable by explainer
            no_classes = len(classifieroutput['intents'])
            
            classes = []
            scores = []

            for i in range(0,4):
                tempscore = classifieroutput['intents'][i]['confidence']
                scores.append(tempscore)
                if return_classes:
                    tempclass = classifieroutput['intents'][i]['intent']
                    classes.append(tempclass)

            if return_classes:
                output = classes + scores
            else:
                output = scores

            return output
        else:
            # classify list and return list
            listlength = len(input_text)
            # TODO: is this 4 because it needs to be fixed for Lime?
            result_array = np.empty((0,4))

            for i in range(0,listlength):
                #print('<' + input_text[i] + '>')
                if not (input_text[i]).isspace():
                    classifieroutput = self.assistant.message(
                        workspace_id=self.workspace_id,
                        input={
                            'text': input_text[i]
                        },
                        alternate_intents=True
                    ).get_result()
                    no_classes = len(classifieroutput['intents'])

                    scores = []

                    for j in range(0,4):
                        tempscore = classifieroutput['intents'][j]['confidence']
                        scores.append(tempscore)
                else:
                    scores = [0] * 4

                result_array = np.vstack((result_array,scores))

            return result_array

    def run(self, input_text, print_results=True):
        output = self.classify_text(input_text, True)
        outputlen = len(output)

        classes = output[0:int((outputlen/2)-1)]
        scores = output[int((outputlen/2)):outputlen-1]

        if print_results:
            print('Intents: ' + '\t'.join(map(str, classes)))
            print('Scores: ' + '\t'.join(map(str, scores)))

        # explain class
        explainer = LimeTextExplainer(class_names=classes)
        exp = explainer.explain_instance(input_text, self.classify_text, num_features=7, top_labels=3, num_samples=100)

        # print explanation
        if print_results:
            print("")
            print('Explanation for class %s' % classes[0])
            print('\n'.join(map(str, exp.as_list(label=0))))

        return exp