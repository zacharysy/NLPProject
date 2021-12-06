from textworld.core import GameState
import torch
import sys
import textworld
sys.path.append("./")
sys.path.append("./translation/")


class BaseAgent(textworld.Agent):
    def act(self, game_state, reward, done):
        return self.callModel(game_state.feedback)

    def callModel(self, text):
        return "open mailbox"


class DQAgent(textworld.Agent):
    def __init__(self, encoder_weights=None, ff_weights=None):
        self.encoder = torch.load(
            encoder_weights) if encoder_weights else encoder_weights
        self.ff = torch.load(
            ff_weights) if ff_weights else ff_weights
        self.state = []

    def act(self, game_state, reward, done):
        while len(self.state) > 1 and self.state[-1].feedback == game_state.feedback:
            self.state.pop()
        self.state.append(game_state)
        return self.callModel(self.state[-1].feedback)

    def callModel(self, action_num):
        # text = rnn.preprocess_line(text)
        # prediction = rnn.predict(self.model, text)
        # return ''.join(prediction)
        return self.ff.action_vocab.denumberize(action_num)


if __name__ == "__main__":
    pass
