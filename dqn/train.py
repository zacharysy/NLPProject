import collections
import textworld
import sys
import torch
import re

from itertools import groupby
from random import choice
from typing import List
from dataclasses import dataclass
from BaseAgent import DQAgent
from random import randint, random
from util import Vocab
from argparse import ArgumentParser


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


@dataclass
class ReplayMemory:
    s_t: str                # state
    a_t: int                # action
    r_t: int                # reward
    s_next: str             # next state
    p_t: int                # priority


def random_sample(iterable, n):
    len_iter = len(iterable)
    return [choice(iterable) for _ in range(n if n <= len_iter else len_iter)]


def mini_sample(rho, replay_memories: List[ReplayMemory]) -> List[ReplayMemory]:
    num_replay_mems = len(replay_memories)
    high_rhos, low_rhos = groupby(replay_memories, lambda x: x.p_t == 1)
    high_rho_ratio = len(list(high_rhos))/num_replay_mems

    if high_rho_ratio <= rho:
        return random_sample(replay_memories, 1000)

    target = int(len(num_replay_mems) * high_rho_ratio)
    train_memories = random_sample(
        high_rhos, target) + random_sample(low_rhos, 1000 - target)
    return train_memories


def main(args):
    word_vocab, action_vocab = read_data()
    dims = 200
    epsilon = 0.1
    rho = 0.25
    gamma = 0.5
    agent = DQAgent(word_vocab,
                    action_vocab, dims)
    replay_mems: List[ReplayMemory] = []

    optim = torch.optim.Adam(agent.dqn.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for _ in range(args.episodes):
        total_loss = 0
        env = textworld.start(zorkPath)
        game_state = env.reset()
        reward, done, moves = 0, False, 0
        desc = ['<CLS>', *preprocess_line(game_state['raw'])]

        for t in range(maxMoves):
            encoding = agent.dqn.encode(desc)
            if random() < epsilon:
                action_num = randint(0, len(action_vocab))
            else:
                dist = agent.dqn(encoding)
                action_num = dist.argmax()

            command = agent.callModel(action_num)
            game_state, reward, done = env.step(command)

            priority = 1 if reward > 0 else 0
            replay_mems.append(ReplayMemory(
                desc, action_num, reward, game_state if not done else 'done', priority))

            train_mems = mini_sample(rho, replay_mems)

            for t in train_mems:
                r = torch.tensor(t.r_t)
                if t.s_next == 'done':
                    y = r
                else:
                    next_enc = agent.dqn.encode(t.s_next)
                    curr_enc = agent.dqn.encode(t.s_t)

                    max_r = torch.max(agent.dqn(next_enc))
                    y = r + (gamma * max_r)
                loss = loss_fn(y, agent.dqn(curr_enc)[t.a_t])

                optim.zero_grad()
                loss.backward()
                optim.step()

                total_loss += loss.detach().item()


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--episodes", type=int,
                   help="number of episodes to train for")
    main(p.parse_args())
