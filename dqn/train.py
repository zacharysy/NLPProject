import textworld
import sys
import torch
import re

from BaseAgent import DQAgent
from random import randint, random
from dqn import DQN
from util import Vocab
from argparse import ArgumentParser


def contains_digit(in_str: str) -> bool:
    """
    This function checks if a string contains any numbers.
    """

    return sum([i.isdigit() for i in in_str]) > 0


def preprocess_line(in_str: str, start_symbol: str = None) -> [str]:
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


sys.path.append("./")
zorkPath = "../benchmark/zork1.z5"
maxMoves = 1000


def read_data():
    words = Vocab()
    actions = Vocab()
    for line in open('./zork_transcript.txt', 'r'):
        if line.startswith('>'):
            actions |= line[1:].strip().split(' ')
        words |= line.strip().split(' ')
    words |= (['<UNK>', '<CLS>'])
    return words, actions


def main(args):
    word_vocab, action_vocab = read_data()
    dims = 200
    epsilon = 0.1
    dqn = DQN(word_vocab, action_vocab, dims)
    agent = DQAgent()

    for _ in range(args.episodes):
        env = textworld.start(zorkPath)
        game_state = env.reset()
        reward, done, moves = 0, False, 0
        desc = ['<CLS>', *preprocess_line(game_state['raw'])]
        for t in range(maxMoves):
            encoding = dqn.encode(desc)
            if random() < epsilon:
                action_num = randint(0, len(action_vocab))
            else:
                dist = dqn(encoding)
                action_num = torch.argmax(dist)
            command = agent.callModel(action_num)
            game_state, reward, done = env.step(command)
            # if reward > 0:


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--episodes", type=int,
                   help="number of episodes to train for")
    main(p.parse_args())
