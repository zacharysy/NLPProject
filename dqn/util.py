from dataclasses import dataclass
from random import choice, sample
import re
from typing import List
from textworld.helpers import start
import torch
import collections
from pprint import pprint


EOS = '<EOS>'
E = 'E'
CLS = "<CLS>"
UNK = "<UNK>"


def contains_digit(in_str: str) -> bool:
    """
    This function checks if a string contains any numbers.
    """

    return sum([i.isdigit() for i in in_str]) > 0


def preprocess_line(in_str: str, start_symbol: str = None) -> List[str]:
    """
    This function is used to preprocess a single line of the input.
    """

    # Add space to punctuation, quotes, and parentheses
    in_str = re.sub('([".,!?()])', r' \1 ', in_str)

    # Strip and split
    line = in_str.lower().split()

    # Filter out any numbers
    line = ['<num>' if contains_digit(i) else i for i in line]

    # Add a start symbol
    if start_symbol is not None:
        line = [start_symbol] + line

    return line


# for debugging


def read_data():
    words = Vocab()
    actions = Vocab()
    v = ""
    for i, line in enumerate(open('./zork_transcript.txt', 'r')):
        if line.startswith('>'):
            actions.add(line[1:].strip())
            v = ' '.join(preprocess_line(v.strip(), start_symbol=CLS))
            v = ""
        else:
            words |= preprocess_line(line, start_symbol=CLS)
            if i != 0:
                v += line

    words |= (['<UNK>', '<CLS>'])
    return words, actions


@dataclass
class ReplayMemory:
    s_t: List[str]          # state
    a_t: int                # action
    r_t: int                # reward
    s_next: List[str]       # next state
    priority: int           # priority


def random_sample(iterable, n):
    num_mems = len(iterable)
    k = num_mems if n > num_mems else n
    return sample(iterable, k)


class ReplayMemoryStore():
    def __init__(self, batch_size, rho, max_size):
        self.store: List[ReplayMemory] = []
        self.batch_size = batch_size
        self.rho = rho
        self.max_size = max_size

    def add(self, r: ReplayMemory):
        if len(self.store) <= self.max_size:
            self.store.append(r)
            return
        n_extra = len(self.store) - self.max_size
        to_remove = random_sample(self.store, n_extra)
        for r in to_remove:
            self.store.remove(r)

    def mini_sample(self):
        num_mems = len(self.store)
        high_rhos, low_rhos = [], []

        for mem in self.store:
            (high_rhos if mem.priority == 1 else low_rhos).append(mem)
        high_rho_ratio = len(list(high_rhos))/num_mems

        if self.batch_size > len(self.store):
            return random_sample(self.store, len(self.store))

        if high_rho_ratio <= self.rho:
            train_mems = high_rhos
            while len(train_mems) < self.batch_size:
                train_mems.append(choice(low_rhos))
            return train_mems

        target = int(len(num_mems) * high_rho_ratio)
        high_rho_mems = random_sample(high_rhos, target)
        low_rho_mems = random_sample(low_rhos, self.batch_size - target)

        return high_rho_mems + low_rho_mems


class Vocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""

    def __init__(self):
        words = set()
        self.num_to_word = list(words)
        self.word_to_num = {word: num for num,
                            word in enumerate(self.num_to_word)}

    def add(self, word):
        if word in self:
            return
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


def generate_actions(correct_action, action_vocab: Vocab):
    options = [
        correct_action, *[choice(action_vocab.num_to_word) for _ in range(3)]
    ]
    return options


def get_device():
    # if torch.cuda.is_available():
    #     return 'cuda'
    return 'cpu'


def progress(iterable):
    import os
    import sys
    if os.isatty(sys.stderr.fileno()):
        try:
            import tqdm
            return tqdm.tqdm(iterable)
        except ImportError:
            return iterable
    else:
        return iterable
