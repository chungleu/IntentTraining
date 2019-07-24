"""
Gets number of times the intent is hit in training and the number of times it is hit in pilot
as the first intent, based on the CSV specified in config. 
"""

import for_csv
from config import *

import os
import pandas as pd
import time
import click

from logging import getLogger
logger = getLogger('intent_size')

@click.command()
@click.argument('topic', nargs=1)
def main(topic):
    logger.info("Getting training data...")
    training_data = for_csv.import_data.get_train_test_data(topic, data_type='train')
    training_count = training_data.groupby('Intent').count()['utterance']
    training_count.index = training_count.index.str.lower()
    training_count = training_count.rename(columns={'utterance': 'training'})

    logger.info("Getting pilot data...")
    csv_path = os.path.join(data_dir, data_file)
    ut = for_csv.utils(topic)
    external_data = ut.import_external_data(csv_path, topic)
    external_count = external_data.groupby(intent1_col).count()['utterance']
    external_count.index = external_count.index.str.lower()
    external_count = external_count.rename(columns={'utterance': 'pilot'})

    external_data_sufficient_conf = external_data[external_data[conf1_col] > 0.4]
    external_count_sufficient_conf = external_data_sufficient_conf.groupby(intent1_col).count()['utterance']
    external_count_sufficient_conf.index = external_count_sufficient_conf.index.str.lower()
    external_count = external_count.rename(columns={'utterance': 'pilot > 0.4'})

    all_count = pd.concat([training_count, external_count, external_count_sufficient_conf], axis=1, join='outer', sort=False).rename(columns={0:'training', 1:'pilot', 'utterance':'pilot > 0.4'}).fillna(0)
    all_count['training/pilot'] = all_count['training'] / all_count['pilot']

    timestr = time.strftime("%Y%m%d-%H%M")
    output_filename = 'intent_count_' + topic + '_' + timestr + '.csv'
    out_path = os.path.join(output_folder, output_filename)
    logger.info("Exporting to CSV at {}".format(out_path))
    all_count.to_csv(out_path)

if __name__ == "__main__":
    main()

