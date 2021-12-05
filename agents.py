from __future__ import annotations
import random, re
import spacy
from pprint import pprint
import translation.rnn as rnn
import translation.transformer as transformer
import torch
import sys
import textworld
sys.path.append("./")
sys.path.append("./translation/")

from baseline.nounverb import NounVerb
from baseline.utils import extract_nouns


class BaseAgent(textworld.Agent):
    def act(self, game_state, reward, done):
        return self.callModel(game_state.feedback)

    def callModel(self, text):
        return "open mailbox"

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

        words = re.sub("(\W|_)+", " ", self.state[-1].feedback).split()
        random.shuffle(words)
        verb = None

        while verb is None:
            word = words.pop()
            verb = self.model(word)

        return f"{verb} {word}"

class TransformerAgent(textworld.Agent):
    """
    More than meets the eye
    """

    def __init__(self, weight_path='./translation/transformer.ckpt'):
        # Load model
        # self.model = transformer.load_model(weight_path)
        self.model = torch.load(weight_path)

    def act(self, game_state, reward, done):
        return self.callModel(game_state.feedback)

    def callModel(self, text):
        # Parse the input text
        text = transformer.preprocess_line(text)

        # Run the prediction
        prediction = transformer.predict(self.model, text)

        return ''.join(prediction)


class RNNAgent(textworld.Agent):
    def __init__(self, weight_path='./translation/rnn.pt'):
        self.model = torch.load(weight_path)

    def act(self, game_state, reward, done):
        return self.callModel(game_state.feedback)

    def callModel(self, text):
        text = rnn.preprocess_line(text)
        prediction = rnn.predict(self.model, text)
        return ''.join(prediction)


if __name__ == "__main__":
    pass
