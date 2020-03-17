import os
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, "data") # ./data
training_dir = os.path.join(data_dir, "workspace_training") # ./data/workspace_training directory containing training CSVs with the naming convention "topic_questions.csv", and test set CSVs with the naming convention "topic_blindset.csv"
output_folder =  os.path.join(current_dir, "results") # ./results

max_workspaces = 20 # for kfold check

### STOPWORDS & CHARS TO IGNORE FOR NGRAM EXTRACTION
# these stopwords will be used on top of the default nltk stopwords (english only) if you select the option 'config'
stopwords = ['hi', 'just', 'hello', 'could', 'might', 'must', 'need', 'would', 'thank', 'thanks' ,"'", ".", ",", "!", ";", "?"] 

# these characters are removed before processing into ngrams
chars_remove = "-:;(),"


# ------------------------------------------------------------------------------------------------------------------------
""" YOU CAN IGNORE EVERYTHING BELOW HERE (unless you plan to use the CLI tools to fetch data from production) """
# ------------------------------------------------------------------------------------------------------------------------

# PARAMS FOR CLEANING EXTERNAL DATA
dfclean_specialchars = ['.', ',', '/', 'pounds', '£', ' ', 'Â', '-', '?', '!', 'today', 'th', 'st', 'gbp', 'and', 'now', '&', 'for', 'the', 'january', 'february', 'march',
                        'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                         'sept', 'sep', 'oct', 'nov', 'dec']

other_drop = []

# PARAMS FOR EXTERNAL DATA (don't need to fill in)
data_file = r"" # filename of external data

responses_todrop_path = r""
buttons_path = r""
follow_up_path = r""
master_failed_responses = r""

utterance_col = 'utterance'
response_col = 'response'
conf1_col = 'confidence1_0'
workspace1_col = 'modelRef_0'
workspace_list = []


# prioritised training
margin_params = {
    'margin_max': 0.5, 
    'min_conf1': 0.2
    }
minhash_params = {
    'threshold': 0.4, 
    'num_perm': 512, 
    'shingle_length': 5
    }
lowconf_max = 0.1
dissimilarity_params = {
    'dissimilarity_min': 0,
    'sortby_margin': False
}

max_conf1 = 0.9