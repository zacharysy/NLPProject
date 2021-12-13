from dataclasses import dataclass
from random import choice
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
    s_t: str                # state
    a_t: int                # action
    r_t: int                # reward
    s_next: str             # next state
    p_t: int                # priority


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
    if torch.cuda.is_available():
        return 'cuda'
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


def read_labels(file):
    # from homework 4
    """Read words and labels from file.

    Parameters:
    - file: file object (not filename!) to read from

    The format of the file should be one sentence per line. Each line
    is of the form

    word1:label1 word2:label2 ...
    """

    ret = []
    for line in file:
        words = []
        labels = []
        for wordlabel in line.split():
            try:
                word, label = wordlabel.rsplit(':', 1)
            except ValueError:
                raise ValueError(f'invalid token {wordlabel}')
            words.append(word)
            labels.append(label)
        ret.append((words, labels))
    return ret
