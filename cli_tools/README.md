# Intent Training Tools


A series of tools to best train classifiers, by getting the best unlabelled utterances, out of scope ngrams, and a single source of truth for training data.

## Contents
### Getting utterances
1. **Get utterances from unlabelled pool** : `get_utterances`. Use arguments [test|train] to get utterances for the specific purpose.

### Getting training utterances
2. **Get utterances containing a phrase** : `get_utterances_containing`. Looks through CSV extract for utterances containing an ngram, which is passed as a command line argument
3. **Get out of scope utterances** : `get_out_of_scope_utterances`. Retrieves utterances that are out of scope (failed twice on master), or ngrams that are in production that don't exist in training

### Summarising what's already in training, and diagnosing issues
4. **Ngrams and their frequencies** : `get_ngram_frequencies`. Retrieves all the ngrams for a specified *n_list* within a workspace and their frequencies, per intent. 
5. **Intent intersections** : `get_intent_intersections`. Returns a matrix of all ngram overlaps between intents for one workspace.
6. **Ngram overlap between two intents** : `get_ngram_overlap`. Returns the ngrams which are common to two intents within a topic (a representation of their semantic overlap), and their frequencies in each intent. 
7. **Diagnosing a misclassification** : `diagnose_confusion`. Given misclassifications from a blind set test, will fetch all the training samples which may have contributed towards these misclassifications, and the ngrams which could be causing this. (very alpha)

### Other utils
8. **Check quality of a test set** : `check_blindset`
9. **Download intents in a workspace as CSV** : `download_intents`. No proxy support currently.
10. **Join external data** : `join_csv_data`. Joins all CSV files in a folder, for data that has been extracted separately.

All of these tools have built in help guides, which can be access using the `--help` flag.

## Installation
To install dependencies run `pip install -r requirements.txt`.