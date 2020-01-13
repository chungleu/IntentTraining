import pandas as pd 
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
def main(input_name, output_name, skill_name):
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
    logger.info("Splitting {} utterances into permutations..".format(df.shape[0]))
    for n in range(df.shape[0]):
        # get the sentence and its intent
        sen = df.iat[n, 0]
        intent = df.iat[n, 1]
        # generate all the different sentences
        res = sentenceSplitter(sen)
        # flatten untill flat (from nested lists to a single list)
        while (any(isinstance(el, list) for el in res)):
            res = flatten(res)
        # Distinguish between single elements and lists
        if (type(res) ==type([])):
            #list
            res = map (lambda x: (x,intent), res)
            outputFrame = pd.DataFrame(res)
            outputFrameP = outputFrameP.append(outputFrame) 
        else:
            #single element
            res = [(res,intent)]
            outputFrameP = outputFrameP.append(res)
            
    # write the frame to CSV without the idexes
    outputFrameP.to_csv(output_path, header=None, index=False)
    logger.info("{} new utterances saved to {}".format(len(outputFrameP), output_path))

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
