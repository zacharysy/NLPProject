import textworld
import torch
from dqn.MA_DQN import MA_DQN


class DQAgent(textworld.Agent):
    def __init__(self, word_vocab, action_vocab, dims, dqn_weights=None):
        self.dqn = MA_DQN(word_vocab, action_vocab,
                          dims) if not dqn_weights else torch.load(dqn_weights)
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
        return self.dqn.action_vocab.denumberize(action_num)
