"""
Set of functions to calculate similarity between two strings

To use from an external script (shingle length optional):
import string_similarity
jaccard_sim = string_similarity.jaccard_similarity_strings(string1, string2, shingle_length)
"""

def extract_shingles(document, shingle_length=5):
    """
    Returns list of shingles of length determined by the input.
    """
    shingles = [document[i:i+shingle_length] for i in range(len(document))][:-shingle_length]

    return shingles

def jaccard_similarity_shingles(shingles, other_shingles):
    """
    Return jaccard similarity of two strings based on input singles.
    This is the size of the set intersection divided by the size of the set union.
    """
    jaccard_sim = len(set(shingles) & set(other_shingles)) / len(set(shingles) | set(other_shingles))

    return jaccard_sim

def jaccard_similarity_strings(string, other_string, shingle_length=5):
    shingles = extract_shingles(string, shingle_length)
    other_shingles = extract_shingles(other_string, shingle_length)

    return jaccard_similarity_shingles(shingles, other_shingles)

def similarity_threshold_minhash_pair(shingles, other_shingles, threshold = 0.4):
    """
    Test of datasketch minhash functionality. Determines whether a pair of strings
    are similar according to the threshold set in the function input.
    """
    from datasketch import MinHash, MinHashLSH

    shingles_set = set(shingles)
    other_shingles_set = set(other_shingles)

    mh = MinHash(num_perm=128)
    oth_mh = MinHash(num_perm=128)

    for sh in shingles_set:
        mh.update(sh.encode('utf8'))

    for sh in other_shingles_set:
        oth_mh.update(sh.encode('utf8'))

    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    lsh.insert("mh", mh)
    result = lsh.query(oth_mh)
    
    if len(result) > 0:
        print('Strings detected as pairs above set threshold of ' + str(threshold))
    else:
        print('Strings not detected as pairs for threshold of ' + str(threshold))
    
if __name__ == '__main__':
    ### TEST
    test_string = r"How can l change from paper statements to paperless statements"
    other_string = r"Can you tell me how I can change from having paper statements to paperless statements"
    print(test_string)
    print(other_string)
    print( jaccard_similarity_strings(test_string, other_string) )

    print('')

    test_string = r"Don't recognize a transaction"
    other_string = r"Unsure transaction"
    print(test_string)
    print(other_string)
    print( jaccard_similarity_strings(test_string, other_string) )

    ### PAIR HASHING TEST
    print('')
    similarity_threshold_minhash_pair(extract_shingles(test_string), extract_shingles(other_string), threshold=0.1)