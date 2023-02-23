import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import collections
import random

WORD_LEN = 8
BATCH_SIZE = 32
VOCAB_SIZE = 65
# N_EMBD = 32
NUM_STEPS = 10000
VALIDATION_FRACTION = 0.8

with open('input.txt', 'r') as f:
    contents = f.read()


def pick_random_batches(contents, vocab_dict, validation=False):
    input = torch.zeros((BATCH_SIZE, WORD_LEN), dtype=torch.int64)
    target = torch.zeros((BATCH_SIZE, WORD_LEN), dtype=torch.int64)
    for i in range(BATCH_SIZE):
        start = random.randint(
            0, int((len(contents) - WORD_LEN - 1) * VALIDATION_FRACTION))
        if validation:
            start = random.randint(
                int(len(contents) * VALIDATION_FRACTION), (len(contents) - WORD_LEN - 1))
        input[i] = torch.tensor([vocab_dict[c]
                                 for c in contents[start:start+WORD_LEN]])
        target[i] = torch.tensor([vocab_dict[c]
                                  for c in contents[start + 1:start + 1 + WORD_LEN]])
    return (input, target)
    # input = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(BATCH_SIZE, 1)
    # target = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).repeat(BATCH_SIZE, 1)
    # return (input, target)


class ValidDataLoaderIterator:
    def __init__(self):
        self.idx = int(len(contents) * VALIDATION_FRACTION)

    def __next__(self):
        if self.idx + 1 + WORD_LEN > len(contents):
            raise StopIteration

        input = torch.zeros((BATCH_SIZE, WORD_LEN), dtype=torch.int64)
        target = torch.zeros((BATCH_SIZE, WORD_LEN), dtype=torch.int64)
        for i in range(BATCH_SIZE):
            input[i] = torch.tensor([vocab_dict[c]
                                     for c in contents[self.idx:self.idx+WORD_LEN]])
            target[i] = torch.tensor([vocab_dict[c]
                                      for c in contents[self.idx + 1:self.idx + 1 + WORD_LEN]])
            self.idx += WORD_LEN
        return (input, target)


class ValidDataLoader:
    def __init__(self):
        pass

    def __iter__(self):
        return ValidDataLoaderIterator()


def get_vocab(contents):
    vocab_dict = {}

    letters = collections.defaultdict(int)
    for c in contents:
        letters[c] += 1
    keys = sorted(letters.keys())
    count = 0
    for k in keys:
        # print('fill', k, count)
        vocab_dict[k] = count
        count += 1
    # print(vocab_dict)
    # print(len(keys))
    return vocab_dict


class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)
        # self.sm = nn.Softmax(dim=2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, target):
        # input is (B, T)
        # output is (B, T, C)
        logits = self.embedding(input)
        loss = self.loss_fn(torch.permute(logits, (0, 2, 1)), target)
        # print('out sum', out[0, 0].sum())
        return logits, loss

# def loss_fn(output, target):
#     target_one_hot = F.one_hot(target, num_classes=VOCAB_SIZE)
#     # print(output.shape, target.shape, target_one_hot.shape)
#     # print((output - target_one_hot)[0,0])
#     # print(output.sum(dim=2))
#     # print('sums', ((output - target_one_hot) ** 2).sum(dim=2))
#     return ((output - target_one_hot) ** 2).sum()


vocab_dict = get_vocab(contents)

model = BigramModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss = None
train_losses = []
valid_losses = []
for i in range(NUM_STEPS):
    train_input, train_target = pick_random_batches(
        contents, vocab_dict, validation=False)
    # print(input.dtype)
    optimizer.zero_grad()
    _output, loss = model(train_input, train_target)
    # print(output.shape)
    # loss = loss_fn(output, target)

    # print(model.parameters())
    loss.backward()
    # for param in model.parameters():
    #     print('grad shape', param.grad.shape)
    #     print('add grad', param.grad)
    optimizer.step()

    valid_input, valid_target = pick_random_batches(
        contents, vocab_dict, validation=True)
    with torch.no_grad():
        _output, valid_loss = model(valid_input, valid_target)
    train_losses.append(loss.item())
    valid_losses.append(valid_loss.item())
print('train loss', loss)

# matplotlib.style.use('default')
plt.plot(train_losses, label='train')
plt.plot(valid_losses, label='valid')
plt.legend()
plt.show()
