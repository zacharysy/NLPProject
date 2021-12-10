from dqn.DQN import DQN
from textworld.core import GameState
import translation.rnn as rnn
import translation.transformer as transformer
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
        self.state = []

    def act(self, game_state, reward, done):
        while len(self.state) > 1 and self.state[-1].feedback == game_state.feedback:
            self.state.pop()
        self.state.append(game_state)
        return self.callModel(self.state[-1].feedback)

    def callModel(self, text):
        text = rnn.preprocess_line(text)
        prediction = rnn.predict(self.model, text)
        return ''.join(prediction)


class DQAgent(textworld.Agent):
    def __init__(self, word_vocab, action_vocab, dims, dqn_weights=None, *args, **kwargs):
        self.dqn = DQN(word_vocab, action_vocab,
                       dims) if not dqn_weights else torch.load(dqn_weights)
        self.state = []
        super().__init__(*args, **kwargs)

    def act(self, game_state, reward, done):
        while len(self.state) > 1 and self.state[-1].feedback == game_state.feedback:
            self.state.pop()
        self.state.append(game_state)
        return self.callModel(self.state[-1].feedback)

    def callModel(self, action_num):
        # text = rnn.preprocess_line(text)
        # prediction = rnn.predict(self.model, text)
        # return ''.join(prediction)
        return self.dqn.action_vocab.denumberize(action_num)


if __name__ == "__main__":
    pass
