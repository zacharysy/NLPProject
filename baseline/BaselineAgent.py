# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


from __future__ import annotations
import random, re, sys, pprint
import spacy

import textworld
import textworld.challenges
from textworld.generator.game import GameOptions
import textworld.generator

sys.path.append("./")
from baseline.nounverb import NounVerb
from baseline.utils import extract_nouns


def get_nouns(text: str) -> [str]:
    # Step 1: filter out header
    lines = [i.strip() for i in text.split('\n')]
    lines = [i for i in lines if len(i) > 0 and i[0] not in '-_|\\']

    # Step 2: split on sentences
    text_new = ' '.join(lines)
    sentences = re.split(r'(?:(?<=[.!?])|(?<=[.!?]["”]))\s+', text_new)

    # Step 3: extract nouns
    out = list(set([j for i in sentences for j in extract_nouns(i)]))

    return out


class BaselineAgent(textworld.Agent):
    def __init__(self):
        # Load noun/verb pairs
        with open('./baseline/nounverb.py') as f:
            data = f.readlines()
        data = list(filter(lambda x: len(x) == 2, map(lambda x: x.strip().split(' '), data)))

        self.model = NounVerb(data)


    def act(self, game_state, reward, done):
        nouns = get_nouns(game_state.feedback)
        noun = random.choice(nouns)
        verb = self.model(noun)

        return f"{verb} {noun}"
