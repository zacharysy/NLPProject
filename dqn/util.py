import torch
import collections
from pprint import pprint


EOS = '<EOS>'
E = 'E'


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


def read_data(path):
    return [(sentence, act) for sentence, act in (line.strip().split('\t')
            for line in open(path))]


def unkify(path):
    counter = collections.Counter()
    for sentence, _ in read_data(path):
        counter.update(sentence.split(' '))


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
