import torch
import pymagnitude
from layers import LinearLayer, SelfAttention
from util import get_device


class Encoder(torch.nn.Module):
    def __init__(self, dims, embedding_path):
        super().__init__()
        self.embeddings = pymagnitude.Magnitude(embedding_path)
        self.fpos = torch.nn.Parameter(
            torch.empty(3*dims, dims, device=get_device()))

        self.sa1 = SelfAttention(dims=dims)
        # self.sa2 = SelfAttention(dims=dims)
        # self.sa3 = SelfAttention(dims=dims)

        self.ll1 = LinearLayer(input_dims=dims, output_dims=dims)
        # self.ll2 = LinearLayer(input_dims=dims, output_dims=dims)
        # self.ll3 = LinearLayer(input_dims=dims, output_dims=dims)

        torch.nn.init.normal_(self.fpos, std=.01, mean=0.0)

    def encode(self, words):
        v = torch.tensor(self.embeddings.query(
            words), device=get_device())
        v += self.fpos[:len(words)]

        # sa1 = self.sa1(v)
        # ll1 = self.ll1(sa1)

        # sa2 = self.sa2(ll1)
        # ll2 = self.ll2(sa2)

        # sa3 = self.sa3(ll2)
        # ll3 = self.ll3(sa3)
        # return ll3[0]
        # return ll1[0]
        # print(torch.mean(v, dim=0).shape)
        return torch.mean(v, dim=0)


class FF(torch.nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.ll1 = LinearLayer(input_dims=dims, output_dims=1)
        self.ll2 = LinearLayer(input_dims=dims, output_dims=1)
        # self.ll3 = LinearLayer(input_dims=dims, output_dims=1)
        # self.relu = torch.nn.ReLU()

    def forward(self, encoding):
        ll1 = self.ll1(encoding)
        # relu1 = self.relu(ll1)
        ll2 = self.ll2(ll1)
        # relu2 = self.relu(ll2)
        # ll3 = self.ll3(relu2)
        # return ll3
        return ll2


class PA_DQN(torch.nn.Module):
    def __init__(self, dims, embedding_path):
        super().__init__()
        self.state_encoder = Encoder(dims, embedding_path)
        self.action_encoder = Encoder(dims, embedding_path)
        self.ff = FF(2 * dims)

    def encode(self, words, action):
        state_enc = self.state_encoder.encode(words)
        action_enc = self.action_encoder.encode(action)
        cat = torch.cat((state_enc, action_enc))
        return cat

    def forward(self, encoding):
        return self.ff.forward(encoding)

    def save(self, filename):
        torch.save(self.state_dict(), filename)
