from math import log
from numpy import array
from numpy import argmax
import numpy as np

# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences


# define a sequence of 10 words over a vocab of 5 words
data = np.random.rand(10, 5)
data = array(data)
# decode sequence
result = beam_search_decoder(data, 4)
# print result
print(data)
for seq in result:
    print(seq)
