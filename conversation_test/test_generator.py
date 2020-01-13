# Original build - Kiril Nedkov; Build for this repo - Kalyan Dutia

import pandas as pd 
import numpy as np
import os
import sys
import time
import click

sys.path.append('..')
import for_csv.logger
from logging import getLogger
logger = getLogger("blindset")

@click.command()
@click.argument('input_name', nargs=1)
@click.option('--output_name', '-o', default=None, help="Specify what to call output file, which will be returned in the output folder specified in config. If not specified, a default is used.")
@click.option('--skill_name', '-s', default=None, help="Specify the skill name. The output file will be named <skill_name>_generator_output_{time}.csv if this is specified, otherwise test_generator_output_{time}.csv.")
@click.option('--sample_limit', '-l', type=int, default=None, help="If used, this will limit the number of records in the expanded test set. Sampling will be done at random.")
def main(input_name, output_name, skill_name, sample_limit):
    import config # data and output dirs
    
    # Read the input file
    input_path = os.path.join(config.data_dir, input_name)
    logger.info("Using input {}".format(input_path))
    df = pd.read_csv(input_path, header=None)
    outputFrameP = pd.DataFrame()

    # create output path
    if output_name is not None and skill_name is not None:
        raise ValueError("Please only specify one of output_name and skill_name.")
    
    skill_name = skill_name or 'test'

    if output_name != None:
        output_path = os.path.join(config.output_folder, output_name)
    else:
        timestr = time.strftime("%Y%m%d-%H%M")
        output_path = os.path.join(config.output_folder, f"{skill_name}_generator_output_{timestr}.csv")

    # for every row in the file
    logger.info("Splitting {} original utterances into permutations..".format(df.shape[0]))
    for n in range(df.shape[0]):
        # get the sentence and its intent
        sen = df.iat[n, 0]
        intent = df.iat[n, 1]
        # generate all the different sentences
        res = sentenceSplitter(sen)
        # flatten until flat (from nested lists to a single list)
        while (any(isinstance(el, list) for el in res)):
            res = flatten(res)
        # Distinguish between single elements and lists
        if (type(res) == type([])):
            #list
            res = map (lambda x: (x,intent), res)
            outputFrame = pd.DataFrame(res)
            outputFrameP = outputFrameP.append(outputFrame) 
        else:
            #single element
            res = [(res,intent)]
            outputFrameP = outputFrameP.append(res)

    # reduce number of samples in final test set, to limit API calls when running tests
    # TODO: better alternative than random?
    if sample_limit is not None:
        # Take stratified sample over intents, returning all samples in the intent if there aren't enough.
        # Make up for the difference by randomly sampling from the remaining records that haven't been chosen yet.
        logger.info("Returned {} utterance in total; reducing to {}".format(len(outputFrameP), sample_limit))

        samples_per_intent = np.floor(outputFrameP[1].value_counts() * sample_limit / len(outputFrameP)).apply(int)
        df_sampled = pd.DataFrame()
        
        for intent in outputFrameP[1].unique():
            df_intent = outputFrameP[outputFrameP[1] == intent]

            if samples_per_intent[intent] > len(df_intent):
                df_sampled = df_sampled.append(df_intent)
            else:
                df_sampled = df_sampled.append(df_intent.sample(samples_per_intent[intent]))

        # make up for rounding errors
        lendiff = sample_limit - len(df_sampled)
        extra_records = pd.concat([outputFrameP, df_sampled]).drop_duplicates(keep=False).sample(lendiff)
        df_sampled = df_sampled.append(extra_records)

        output_df = df_sampled

    else:
        output_df = outputFrameP
                        
    # write the frame to CSV without the idexes
    output_df.to_csv(output_path, header=None, index=False)
    logger.info("{} new utterances saved to {}".format(len(output_df), output_path))

# Flatten function
flatten = lambda l: [item for sublist in l for item in sublist]

# function that returns generated tests from 1 sentence
def sentenceSplitter(sen):
    """
    Return a series of test cases from one utterance which specifies permutations using the square bracket syntax
    "I am an [example/test/pretend] utterance".
    """
    #check if there is things to split
    if ("[" not in sen):
        return sen

    #split
    left = sen.find("[")
    right= sen.find("]")
    head = sen[0:left]
    body = sen[left+1:right].split("/")
    tail = sen[right+1:len(sen)]

    #generate the cases
    mas = []
    for part in body:
        sen1 = head + part + tail
        mas.append(sentenceSplitter(sen1))
    return mas

main()
