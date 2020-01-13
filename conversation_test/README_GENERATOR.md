This file explains the use of the following python scripts:

1. test_generator.py
    - used to procedurally generate test cases from user formatted .csv file 
    - The input format should be a .csv file that has 2 columns (utterance , intent), e.g. 
                            Sentence 1  , Intent1
                            Sentence 2  , Intent2
                            Sentence 3  , Intent3
    - The script takes any text inside of [] brackets and separates it based on / e.g. 

                    Sentence [1/2/3/4]  , Intent1            will become    

                            Sentence 1  , Intent1
                            Sentence 2  , Intent1
                            Sentence 3  , Intent1
                            Sentence 4  , Intent1
    - Also works for complex sentences (2 or more [])

        Sentence [1/2/3/4] [true/false] , Intent1            will become 

                        Sentence 1 true , Intent1
                        Sentence 2 true , Intent1
                        Sentence 3 true , Intent1
                        Sentence 4 true , Intent1
                       Sentence 1 false , Intent1
                       Sentence 2 false , Intent1
                       Sentence 3 false , Intent1
                       Sentence 4 false , Intent1
    - The script is deterministic
    - Usage example    
        test_generator.py <inputFile>.csv <outputfile>.csv
            or
        test_generator.py <inputFile>.csv
    - The input file needs to exist while the output file doesn't (be careful the script will override the output file without a prompt!)

2. test_simplifier.py
    - used to create test files of manageble size while trying to keep the overall results
    - The input format should be a .csv file that has 2 columns (utterance , intent), e.g. 
                            Sentence 1   , Intent1
                            Sentence 2   , Intent2
                            Sentence 3   , Intent3
    - The user is asked to specify a maximum number of cases per intent (n)
    - If an intent has less than the specified number (n) of utterances the script includes all of them 
    - If an intent has more than the specified number (n) of utterances the sctipt includes n of them at random
    - The script is nondeterministic
    - Usage example
        test_simplifier.py <inputFile>.csv <outputfile>.csv
            or
        test_simplifier.py <inputFile>.csv
    - The input file needs to exist while the output file doesn't (be careful the script will override the output file without a prompt!)