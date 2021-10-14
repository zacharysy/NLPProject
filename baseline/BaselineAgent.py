# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


from __future__ import annotations
import random, re, sys, pprint
import spacy

from pprint import pprint

import textworld
import textworld.challenges
from textworld.generator.game import GameOptions
import textworld.generator

sys.path.append("./")
from baseline.nounverb import NounVerb
from baseline.utils import extract_nouns

class BaselineAgent(textworld.Agent):
    def __init__(self):
        # Load noun/verb pairs
        with open('./baseline/pairs.txt') as f:
            data = f.readlines()
        data = list(filter(lambda x: len(x) == 2, map(lambda x: x.strip().split(' '), data)))

        self.model = NounVerb(data)

        # Keep a memory of the last few game states
        self.state = []

    def act(self, game_state, reward, done):
        # Check if the game state stayed the same
        self.state.append(game_state)
        while self.state[-1].feedback == game_state.feedback and len(self.state) > 1:
            self.state.pop()

        words = self.state[-1].feedback.split()
        random.shuffle(words)
        verb = None

        while verb is None:
            word = words.pop()
            verb = self.model(word)

        return f"{verb} {word}"

if __name__ == "__main__":
    agent = BaselineAgent()
    env = textworld.start("zork1.z3")
    game_state = env.reset()
    reward, done, moves = 0, False, 0

    while not done and moves < 1000:
        print(game_state.feedback)
        command = agent.act(game_state, reward, done)
        game_state, reward, done = env.step(command)

        print("> ", command)

        moves += 1

