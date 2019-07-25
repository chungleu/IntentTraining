"""
Get utterances containing a string that is passed in as the only command line argument. Searches
through the pilot logs CSV specified in config.py.
"""
# TODO: switch to using logger
import sys
sys.path.append('..')

from config import *
import for_csv
import click
import pandas as pd
import os, time

@click.command()
@click.argument('string_arg', nargs=1, type=str)
@click.option('--topic', '-t', type=str, default=None)
@click.option('--training', '-tr', is_flag=True)
@click.option('--case_sensitive', '-cs', is_flag=True)
@click.option('--just_utterances', '-u', is_flag=True)
@click.option('--to_csv', '-csv', is_flag=True)
def get_utterances_containing_main(string_arg, topic, case_sensitive, just_utterances, to_csv, training):
    """
    Used so that the child function can be imported.
    """
    return get_utterances_containing(string_arg, topic, case_sensitive, just_utterances, to_csv, training)

def get_utterances_containing(string_arg, topic, case_sensitive, just_utterances, to_csv, training, hide_output=False):
    
    string_arg = str(string_arg)
    if not case_sensitive:
        string_arg = string_arg.lower()
    
    if training:
        from get_out_of_scope_utterances import get_training_data
        df = get_training_data(topic)

        pdf = for_csv.process_dataframe(df, utterance_col='utterance')
        workspace_col = 'topic'
        intent_col = 'Intent'
    else:
        from get_out_of_scope_utterances import get_external_data
        df = get_external_data(topic)

        pdf = for_csv.process_dataframe(df, utterance_col=utterance_col, conf1_col=conf1_col)
        workspace_col = workspace1_col
        intent_col = intent1_col

    rows_containing_string = pdf.get_utterances_containing_string(string_arg, lower=not(case_sensitive))

    if to_csv:
        if not hide_output:
            print('Summary:')
            print(rows_containing_string.groupby([workspace_col, intent_col]).count()[utterance_col])
        timestr = time.strftime("%Y%m%d-%H%M")
        filename = 'utterances_containing_' + string_arg + '_' + timestr + '.csv'
        file_path = os.path.join(output_folder, filename)
        rows_containing_string.to_csv(file_path, index=False)
    elif just_utterances:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'max_colwidth', 1000):
            if not hide_output:
                print(rows_containing_string[pdf.utterance_lower_col])
    else:
        if not hide_output:
            print(rows_containing_string)

    return rows_containing_string

if __name__ == "__main__":
    get_utterances_containing_main()