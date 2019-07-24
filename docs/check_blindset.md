# `check_blindset`: ensuring test set quality

The `check_blindset.py` script takes paths of a training and test set for a workspace as arguments, and performs the following checks in order:

1. All the intent names used in the test set are intent names used in the bot.
2. Duplicates in the test set:
   a. Drops any duplicates in both utterance & intent
   b. Flags duplicate utterances with different intents, in the `duplicate_utterance` column of the final CSV.
3. Duplicates between test and train:
   a. Drops any [utterance, intent] records in test that exist in train
   b. (TODO) Flags any utterances in test that are filed under a different intent in train.
4. Flags any utterances in test which have a very similar counterpart in training, in the `similar_to_training` column of the final CSV.
5. Checks overall test set size is greater than 25% (default) of the training size.
6. Checks whether the test set size for each intent is greater than the 20% (default) of the training size. Intents that need more data, and the amount of data they need are outputted to *results/test_set_intents_to_small.csv*.

It then exports the test set, with all duplicates removed and columns detailing the above recommendations, to `results/test_set_recommendations.csv`.