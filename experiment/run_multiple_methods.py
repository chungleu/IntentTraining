"""
Run multiple methods / workspaces at once
"""
from run_test import run_test
from summarise_results import summarise_results

def run_method_workspace(method, topic, orig_training_size=0.1, no_iterations=7, train_blind_split=0.8):
    rt = run_test(topic, method, orig_training_size=orig_training_size, no_iterations=no_iterations, train_blind_split=train_blind_split)

    intentpath = 'data/'+topic+'_questions.csv'
    blindsetpath = 'data/'+topic+'_blindset.csv'
    questions = rt.import_data(intentpath)
    blindset = rt.import_data(blindsetpath)
    #blindset = blindset.iloc[1:30] # DEBUG

    intial_questions_train, initial_questions_test = rt.train_test_split(questions, orig_training_size)

    workspace_list = rt.post_workspace(intial_questions_train)
    print(workspace_list)

    rt.iteratively_add_test(workspace_list, intial_questions_train, initial_questions_test, blindset)

    summarise = summarise_results(workspace_conf_thresh=0.4, method=method, topic=topic)
    joined_df = summarise.join_results_individual_method(method)
    summarise.calculate_stats_per_intent_all_iterations(joined_df, export_csv=True)

    
### RUN
topic = 'fallback'
orig_training_size = 0.99 # default 0.1
no_iterations = 3 # default ?
train_blind_split = 0.5

method_list = ['random', 'margin', 'similarity']
method_list = ['margin']

for method in method_list:
    print('Running ' + topic + ' & ' + method)
    run_method_workspace(method, topic, orig_training_size=orig_training_size, no_iterations=no_iterations)
