"""
Joins all CSV data in the directory specified in config.
"""

# external
import pandas as pd
import os
from datetime import datetime as dt
from tqdm import tqdm

# internal
import for_csv
from config import split_data_dir, data_dir

from logging import getLogger
logger = getLogger('join_csv_data')

def import_and_join(data_dir, first_date_min=20190312, debug=False):
    logger.info('Finding files...')

    csv_files = pd.DataFrame([os.path.splitext(f)[0] for f in os.listdir(data_dir) if (f.endswith('.csv') & f.startswith('divr') & (len(f)<=55))], columns=['file'])
    
    split_csv = csv_files['file'].str.split('_').to_list()

    date_split_csv = []
    for f in split_csv:
        date_split_2_csv = []
        for word in f:
            try:
                date_split_2_csv.append(dt.strftime(pd.to_datetime(word), "%Y%m%d"))
            except ValueError:
                pass
        date_split_csv.append(date_split_2_csv)

    # Concatenate date pairs to dataframe for each file name
    dates_csv = pd.DataFrame(date_split_csv)
    csv_files = pd.concat([csv_files[:], dates_csv[:]], axis = 1)
    csv_files[[0,1]] = csv_files[[0,1]].astype(int)

    csv_files = csv_files[(csv_files[0] >= first_date_min)]
    end_date = csv_files.loc[:, 1].max()
    date_range = [first_date_min, end_date]

    if debug: 
        logger.debug(csv_files)

    logger.info('Importing all files in the folder')
    csv_data = pd.DataFrame()
    for file_name in tqdm(csv_files['file']):
        file_name += '.csv'
        file_path = os.path.join(data_dir, file_name)
        tempdf = pd.read_csv(file_path, low_memory=False)
        csv_data = pd.concat([csv_data, tempdf], sort = False)

    return csv_data, date_range

def export_to_csv(csv_data, export_dir, date_range):
    file_name = 'divr-replica_conversations_{}_to_{}.csv'.format(date_range[0], date_range[1])
    output_path = os.path.join(export_dir, file_name)
        
    logger.info("Exporting file to {}..".format(output_path))
    csv_data.to_csv(output_path)
    logger.info("..done")

if __name__ == '__main__':
    debug = False
    csv_data, date_range = import_and_join(split_data_dir, debug=debug)

    export_to_csv(csv_data, data_dir, date_range)
    