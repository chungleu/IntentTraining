import sys
sys.path.append('..')

# vars: config
from config import *

# libraries
import click

# internal
import for_csv

@click.command()
@click.argument('trainortest', nargs=1, type=click.Choice(['train', 'test']))
@click.argument('topic', nargs=1)
@click.option('--method', '-m', type=click.Choice(['similarity', 'margin', 'lowconf']), default='similarity')
@click.option('--no_utterances', '-n', type=int, prompt=True)
@click.option('--intents', '-i', help='Comma-separated list (no spaces) of specific intents to retrieve.', default=None)

def get_utterances(trainortest, method, topic, no_utterances, intents):
    import os, importlib, time
    # TODO: don't need to prompt for no_utterances if test option used.

    utils = for_csv.utils(topic, margin_params=margin_params, minhash_params=minhash_params, lowconf_max=lowconf_max)

    # import external lists
    responses_todrop = utils.import_csv_to_list(responses_todrop_path)
    button_clicks = utils.import_csv_to_list(buttons_path).tolist()
    follow_up_questions = utils.import_csv_to_list(follow_up_path).tolist()
    utterances_todrop = button_clicks + other_drop + follow_up_questions

    # import data
    print('Importing external data...')
    csv_path = os.path.join(data_dir, data_file)
    df_external = utils.import_external_data(csv_path, topic)

    print('Importing training data for topic ' + topic + '...')
    file_name = topic + '_questions.csv'

    training_path = os.path.join(training_dir, file_name)
    df_training = utils.import_training_data(training_path)

    # clean training data
    df_training = utils.check_questions_df_consistency(df_training, to_lower=False)

    # clean external data (remove button clicks, dates, ..?)
    # TODO: better management of column names. Mapping in dict, in config?
    process = for_csv.process_dataframe(df_external, utterance_col=utterance_col, conf1_col=conf1_col)
    process.remove_numeric_utterances(chars_toignore=dfclean_specialchars)
    process.remove_date_utterances()
    if trainortest == 'train':
        process.drop_confidence_greaterthan(max_conf1)
    process.drop_rows_with_column_value(utterance_col, utterances_todrop)
    process.drop_duplicate_utterances(duplicate_thresh=1)
    process.drop_rows_with_column_value(response_col, responses_todrop, lower=False)
    df_external = process.get_df()

    # filter by intent (optional)
    if intents:
        intents_list = for_csv.process_list_argument(intents, val_type=str)
        df_external = utils.df_select_specific_intents(df_external, intents_list, include_second_intent=True)
        print('Filtered by intents ' + str(intents_list))

    # select utterances
    if trainortest == 'train':
        print('Retrieving utterances for training using method ' + method + '...')
        priority_utterances = utils.get_priority_utterances(no_utterances, df_external, df_training, method=method)
    elif trainortest == 'test':
        print("Dropping utterances that exist in training...")
        utterances_in_train = df_training[utterance_col].tolist()
        # new class instance as df_external may have been filtered by intent
        process2 = for_csv.process_dataframe(df_external, utterance_col=utterance_col, conf1_col=conf1_col)
        process2.drop_rows_with_column_value(utterance_col, utterances_in_train)
        priority_utterances = process.get_df()

    # export to csv
    timestr = time.strftime("%Y%m%d-%H%M")
    base_filename = trainortest + '_candidates_' + topic + '_'
    if intents:
        base_filename += intents + '_'

    if trainortest == 'train':
        base_filename += method + '_'
        
    output_filename = base_filename + timestr + '.csv'
    out_path = os.path.join(output_folder, output_filename)
    priority_utterances.to_csv(out_path, index=False)

if __name__ == '__main__':
    get_utterances()