from typing import List
import torch
from layers import Embedding, LinearLayer, SelfAttention
from util import get_device, Vocab, UNK


class Encoder(torch.nn.Module):
    def __init__(self, word_vocab: Vocab, dims: int):
        super().__init__()
        self.emb = Embedding(vocab_size=len(word_vocab), output_dims=dims)
        self.fpos = torch.nn.Parameter(
            torch.empty(dims, dims, device=get_device()))

        self.sa1 = SelfAttention(dims=dims)
        self.sa2 = SelfAttention(dims=dims)
        self.sa3 = SelfAttention(dims=dims)

        self.ll1 = LinearLayer(input_dims=dims, output_dims=dims)
        self.ll2 = LinearLayer(input_dims=dims, output_dims=dims)
        self.ll3 = LinearLayer(input_dims=dims, output_dims=dims)

        self.word_vocab = word_vocab

        torch.nn.init.normal_(self.fpos, std=.01, mean=0.0)

    def encode(self, words: List[str]):
        word_nums = torch.tensor([self.word_vocab.numberize(
            w if w in self.word_vocab else UNK) for w in words], device=get_device())
        v = self.emb(word_nums) + self.fpos[:len(word_nums)]

        sa1 = self.sa1(v)
        ll1 = self.ll1(sa1)

        sa2 = self.sa2(ll1)
        ll2 = self.ll2(sa2)

        sa3 = self.sa3(ll2)
        ll3 = self.ll3(sa3)
        return ll3[0]


class FF(torch.nn.Module):
    def __init__(self, num_actions: int, dims: int):
        super().__init__()
        self.ll1 = LinearLayer(input_dims=dims, output_dims=dims)
        self.ll2 = LinearLayer(input_dims=dims, output_dims=dims)
        self.ll3 = LinearLayer(input_dims=dims, output_dims=num_actions)
        self.relu = torch.nn.ReLU()

    def forward(self, encoding):
        ll1 = self.ll1(encoding)
        relu1 = self.relu(ll1)
        ll2 = self.ll2(relu1)
        relu2 = self.relu(ll2)
        ll3 = self.ll3(relu2)
        return ll3


class MA_DQN(torch.nn.Module):
    def __init__(self, word_vocab: Vocab, action_vocab: Vocab, dims: int):
        super().__init__()
        self.word_vocab = word_vocab
        self.action_vocab = action_vocab
        self.dims = dims

        self.encoder = Encoder(word_vocab,  dims)
        self.FF = FF(len(action_vocab), dims)

    def encode(self, words):
        return self.encoder.encode(words)

    def forward(self, encoding):
        return self.FF(encoding)
