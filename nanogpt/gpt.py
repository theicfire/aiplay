import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import collections
import random
import matplotlib.pyplot as plt
import matplotlib

WORD_LEN = 8
BATCH_SIZE = 32
VOCAB_SIZE = 65  # Where did 32 come from for C?
# N_EMBD = 32
NUM_STEPS = 1000
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


DMODEL = 256
DHEAD = 256
DROPOUT_PERC = 0.2


class Attention(nn.Module):
    def __init__(self, dhead):
        super().__init__()
        self.k = nn.Linear(DMODEL, dhead)
        self.q = nn.Linear(DMODEL, dhead)
        self.v = nn.Linear(DMODEL, dhead)
        self.dropout = nn.Dropout(DROPOUT_PERC)
        self.register_buffer('tril', torch.tril(
            torch.ones(WORD_LEN, WORD_LEN)))
        self.dhead = dhead  # TODO buffer?

    def forward(self, x):
        # x is (B, T, DMODEL)
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        mul1 = q @ k.transpose(-2, -1)  # or -1, -2.. same!
        mul1 = mul1 / (self.dhead ** 0.5)
        wei = mul1.masked_fill(self.tril[:] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        mul2 = wei @ v  # TODO am I multiplying correctly given tril?
        # return (B, T, dhead)
        return mul2


class MultiAttention(nn.Module):
    def __init__(self, n_heads=4):
        super().__init__()
        self.n_heads = n_heads  # TODO make the gradient here not be tracked
        dhead = DHEAD // n_heads  # TODO confirm this is an even division
        self.heads = [Attention(dhead) for i in range(n_heads)]
        self.fc = nn.Linear(DHEAD, DMODEL)
        self.dropout = nn.Dropout(DROPOUT_PERC)

    def forward(self, x):
        # x is (B, T, DMODEL)
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # return (B, T, DMODEL)
        return self.dropout(self.fc(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DMODEL, 4 * DMODEL)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4 * DMODEL, DMODEL)
        self.dropout = nn.Dropout(DROPOUT_PERC)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = FeedForward()
        self.multi_attention = MultiAttention()
        self.ln1 = nn.LayerNorm(DMODEL)
        self.ln2 = nn.LayerNorm(DMODEL)

    def forward(self, x):
        x = x + self.multi_attention(self.ln1(x))
        return x + self.ff(self.ln2(x))


NUM_BLOCKS = 4


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB_SIZE, DMODEL)
        self.pos_encoding = torch.nn.Embedding(WORD_LEN, DMODEL)
        self.fc = nn.Linear(DMODEL, VOCAB_SIZE)
        # self.sm = nn.Softmax(dim=2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.blocks = [Block() for _ in range(NUM_BLOCKS)]
        self.ln = nn.LayerNorm(DMODEL)

    def forward(self, input, target, training=True):
        # input is (B, T)
        # output is (B, T, C)
        x = self.embedding(input)
        pos = torch.zeros_like(input)
        pos[:] = torch.tensor(range(WORD_LEN), dtype=torch.int64)
        pos = self.pos_encoding(pos)
        x = x + pos
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        logits = self.fc(x)
        loss = None
        if training:
            loss = self.loss_fn(torch.permute(logits, (0, 2, 1)), target)

        # print('out sum', out[0, 0].sum())
        return logits, loss

    def generate(self, vocab_dict, length):
        # input is simply (T)
        # output is (T, C)
        current = torch.zeros((1, WORD_LEN), dtype=torch.int64)
        ret = []
        vocab_dict_reverse = {v: k for k, v in vocab_dict.items()}
        for i in range(length):
            logits, _loss = self.forward(current, target=None, training=False)
            # idx = logits[0][WORD_LEN - 1].argmax().item() # Pick stochastically instead
            idx = torch.multinomial(
                F.softmax(logits[0][WORD_LEN - 1], dim=0), 1).item()

            ret.append(vocab_dict_reverse[idx])
            current = torch.cat([current[:, 1:], torch.tensor([[idx]])], dim=1)

        logits = self.embedding(current)
        # print('logits', logits[0][WORD_LEN - 1])
        # print('softmax', F.softmax(logits[0][WORD_LEN - 1], dim=0))
        return ''.join(ret)

# def loss_fn(output, target):
#     target_one_hot = F.one_hot(target, num_classes=VOCAB_SIZE)
#     # print(output.shape, target.shape, target_one_hot.shape)
#     # print((output - target_one_hot)[0,0])
#     # print(output.sum(dim=2))
#     # print('sums', ((output - target_one_hot) ** 2).sum(dim=2))
#     return ((output - target_one_hot) ** 2).sum()


vocab_dict = get_vocab(contents)

model = GPT()
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
# plt.show()

print(model.generate(vocab_dict, 100))
