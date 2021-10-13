# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import random, re, sys, pprint

import textworld
import textworld.challenges
from textworld.generator.game import GameOptions
import textworld.generator

sys.path.append("./")
from baseline.nounverb import NounVerb

class BaselineAgent(textworld.Agent):
    def __init__(self):
        self.model = NounVerb(data=[("a", "b")])

    def get_nouns(self, text):
        print(text)
        print()
        lines = re.split(r'(?:(?<=[.!?])|(?<=[.!?]["”]))\s+', text)
        pprint.pprint(lines)

    def act(self, game_state, reward, done):
        nouns = self.get_nouns(game_state.feedback)
        noun = random.choice(nouns)
        verb = self.model(noun)

        return f"{verb} {noun}"
