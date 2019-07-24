data_dir = r"" # directory of external (production data)
data_file = r"" # filename of external data
training_dir = r"" # default ./data - directory containing training CSVs with the naming convention "topic_questions.csv", and test set CSVs with the naming convention "topic_blindset.csv"
output_folder =  r"" # default ./results

# PARAMS FOR PRIORITISED TRAINING
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

# PARAMS FOR CLEANING EXTERNAL DATA
dfclean_specialchars = ['.', ',', '/', 'pounds', '£', ' ', 'Â', '-', '?', '!', 'today', 'th', 'st', 'gbp', 'and', 'now', '&', 'for', 'the', 'january', 'february', 'march',
                        'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                         'sept', 'sep', 'oct', 'nov', 'dec']

other_drop = []

# these are CSVs on the shared drive
responses_todrop_path = r""
buttons_path = r""
follow_up_path = r""
master_failed_responses = r""

utterance_col = 'utterance'
response_col = 'response'
conf1_col = 'confidence1_0'
workspace1_col = 'modelRef_0'
workspace_list = []

### STOPWORDS & CHARS TO IGNORE FOR NGRAM EXTRACTION
stopwords = ['hi', 'just', 'hello', 'could', 'might', 'must', 'need', 'would', 'thank', 'thanks' ,"'", ".", ",", "!", ";", "?"] 
chars_remove = "-:;(),"