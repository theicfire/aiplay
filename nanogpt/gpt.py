import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import collections
import random

WORD_LEN = 8
BATCH_SIZE = 4
VOCAB_SIZE = 39  # TODO not 32?
vocab_dict = {}


def pick_random_batches(contents):
    ret = torch.zeros((BATCH_SIZE, WORD_LEN))
    for i in range(BATCH_SIZE):
        start = random.randint(0, len(contents) - WORD_LEN)
        ret[i] = torch.tensor([vocab_dict[c.lower()]
                              for c in contents[start:start+WORD_LEN]])
    return ret


def fill_vocab(contents):
    global vocab_dict
    letters = collections.defaultdict(int)
    for c in contents:
        letters[c.lower()] += 1
    keys = sorted(letters.keys())
    count = 0
    for k in keys:
        # print('fill', k, count)
        vocab_dict[k] = count
        count += 1
    # print(vocab_dict)
    # print(len(keys))


with open('input.txt', 'r') as f:
    contents = f.read()

fill_vocab(contents)
batch = pick_random_batches(contents)
print(batch)

# read random 8-letter sections from the file
