# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim

import re
import math
import tqdm
import random
import collections
import numpy as np

# Behavior modifying constants
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = 'transformer.ckpt'
train_path = '../training/train.txt'
load_weights = False
train_model = True
separate_vocab = True

# Hyperparameters
lr = 0.0003
epochs = 2

# Objects
class Vocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""
    def __init__(self):
        words = {'<EOS>', '<UNK>'}
        self.num_to_word = list(words)
        self.word_to_num = {word:num for num, word in enumerate(self.num_to_word)}
    def add(self, word):
        if word in self: return
        num = len(self.num_to_word)
        self.num_to_word.append(word)
        self.word_to_num[word] = num
    def discard(self, word):
        raise NotImplementedError()
    def __contains__(self, word):
        return word in self.word_to_num
    def __len__(self):
        return len(self.num_to_word)
    def __iter__(self):
        return iter(self.num_to_word)

    def numberize(self, word):
        """Convert a word into a number."""
        if word in self.word_to_num:
            return self.word_to_num[word]
        else:
            return self.word_to_num['<UNK>']

    def denumberize(self, num):
        """Convert a number into a word."""
        return self.num_to_word[num]


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):
    """
    This class implements a transformer model for text-to-command
    generation in a text adventure.

    Inputs: vocab_size - number of words in the vocabulary
            dim_model - size of embedded vectors
            num_heads - number of attention heads
            num_encoder_layers - number of encoder layers
            num_decoder_layers - number of decoder layers

    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    def __init__(self,
                 scene_vocab: Vocab,
                 command_vocab: Vocab,
                 dim_model: int = 100,
                 num_heads: int = 5,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 dropout_p: float = 0.1
        ):

        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.scene_vocab = scene_vocab
        self.command_vocab = command_vocab
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.scene_embedding = nn.Embedding(len(self.scene_vocab), dim_model)
        self.command_embedding = nn.Embedding(len(self.command_vocab), dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, len(self.command_vocab))
        self.softmax = nn.Softmax()

    def encode_src(self, src: [str]):
        return torch.tensor([self.scene_vocab.numberize(i) for i in src], dtype=torch.long).unsqueeze(0)

    def encode_tgt(self, tgt: [str]):
        return torch.tensor([self.command_vocab.numberize(i) for i in tgt], dtype=torch.long).unsqueeze(0)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.scene_embedding(src) * math.sqrt(self.dim_model)
        tgt = self.command_embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        soft_out = self.softmax(out)

        # Permute the output to size (batch_size, out_vocab_size, sequence_len)
        soft_out = soft_out.permute(1, 2, 0)

        return soft_out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)


# Helper functions
def contains_digit(in_str: str) -> bool:
    """
    This function checks if a string contains any numbers.
    """

    return sum([i.isdigit() for i in in_str]) > 0


def preprocess_line(in_str: str) -> [str]:
    """
    This function is used to preprocess a single line of the input.
    """

    # Add space to punctuation, quotes, and parentheses
    in_str = re.sub('([".,!?()])', r' \1 ', in_str)

    # Strip and split
    line = in_str.split()

    # Filter out any numbers
    line = ['<num>' if contains_digit(i) else i for i in line]

    return line


def load_data(path: str) -> [([str], [str])]:
    """
    This function loads in data from the training file. The file
    alternates between scene descriptions and commands.
    """

    # Read the text file
    with open(path, 'r') as f:
        data = list(f.readlines())

    # Parse each file
    data = [preprocess_line(line) for line in data]

    # Split into input/target
    inputs = [line for i, line in enumerate(data) if i % 2 == 0]
    targets = [line for i, line in enumerate(data) if i % 2 == 1]
    data = list(zip(inputs, targets))

    return data


if __name__ == '__main__':
    # Load the data
    data = load_data(train_path)

    # Create vocabularies
    scene_vocab = Vocab()
    command_vocab = Vocab()

    for scene, command in data:
        scene_vocab |= scene
        command_vocab |= command

    if not separate_vocab:
        scene_vocab |= command_vocab
        command_vocab |= scene_vocab

    # Load model
    if load_weights:
        model = torch.load(save_path)
    else:
        model = Transformer(scene_vocab=scene_vocab,
                            command_vocab=command_vocab,
                            dim_model=128,
                            num_heads=4,
                            num_encoder_layers=3,
                            num_decoder_layers=3,
                            dropout_p=0.1)
    model = model.to(device)

    # Train model
    if train_model:
        # Create optimizer and loss function
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            # Perform pre-loop accounting
            random.shuffle(data)
            train_loss = 0

            for scene, command in tqdm.tqdm(data):
                # Encode the vectors
                scene_nums = model.encode_src(scene).to(device)
                command_nums = model.encode_tgt(command).to(device)

                # Forward pass
                out = model(scene_nums, command_nums)
                loss_val = loss_fn(out, command_nums)

                # Backprop
                optim.zero_grad()
                loss_val.backward()
                optim.step()

                # Perform housekeeping
                train_loss += loss_val.detach().item()

        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss}')
        torch.save(model, save_path)
