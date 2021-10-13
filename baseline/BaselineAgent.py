# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import random

import textworld
import textworld.challenges
from textworld.generator.game import GameOptions
import textworld.generator

from nounverb import NounVerb

class BaselineAgent(textworld.Agent):
    def __init__(self):
        self.model = NounVerb(data: [("a", "b")])

    def get_nouns(self, text):
        pass

    def act(self, game_state, reward, done):
        nouns = get_nouns(game_state.feedback)
        noun = random.choice(nouns)
        verb = self.model(noun)

        return f"{verb} {noun}"
