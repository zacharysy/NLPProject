import abc
from random import choice, randint, random
import textworld
import torch
from MA_DQN import MA_DQN
from PA_DQN import PA_DQN
from util import ReplayMemory, Vocab


class DQAgent(textworld.Agent, abc.ABC):
    @abc.abstractmethod
    def explore(self): pass

    @abc.abstractmethod
    def exploit(self): pass

    def act(self, game_state, reward, done):
        while len(self.state) > 1 and self.state[-1].feedback == game_state.feedback:
            self.state.pop()
        self.state.append(game_state)
        return self.callModel(self.state[-1].feedback)

    @abc.abstractmethod
    def callModel(self): pass

    @abc.abstractmethod
    def loss_args(self): pass

    def save(self, filename):
        return torch.save(self.dqn, filename)

    def set_epsilon(self):
        self.epsilon = self.init_epsilon


class MA_DQAgent(DQAgent):
    def __init__(self, word_vocab, action_vocab, dims, dqn_weights=None,
                 init_epsilon=1, end_epsilon=0.2, rho=0.25, gamma=0.5, max_moves=1000):
        self.dqn = MA_DQN(word_vocab, action_vocab,
                          dims) if not dqn_weights else torch.load(dqn_weights)
        self.word_vocab = word_vocab
        self.action_vocab = action_vocab
        self.dims = dims
        self.state = []
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon
        self.end_epsilon = end_epsilon
        self.rho = rho
        self.gamma = gamma
        self.max_moves = max_moves

    def explore(self):
        return randint(0, len(self.action_vocab))

    def exploit(self, enc):
        return self.dqn(enc).argmax()

    def callModel(self, text):
        if random() < self.epsilon:
            action_num = self.explore()
        else:
            enc = self.dqn.encode(text)
            action_num = self.exploit(enc)
        self.epsilon -= (self.init_epsilon - self.end_epsilon) / \
            self.max_moves
        return self.action_vocab.denumberize(action_num)

    def loss_args(self, r, t: ReplayMemory):
        action_num = self.action_vocab.numberize(t.a_t)
        next_enc = self.dqn.encode(t.s_next)
        curr_enc = self.dqn.encode(t.s_t)
        max_r = torch.max(self.dqn(next_enc))
        y = r + (self.gamma * max_r)
        return y, self.dqn(curr_enc)[action_num]


class PA_DQAgent(DQAgent):
    def __init__(self, dims, embedding_path, action_vocab: Vocab, dqn_weights=None,
                 init_epsilon=1, end_epsilon=0.2, rho=0.25, gamma=0.5, max_moves=1000):
        self.dqn = PA_DQN(
            dims, embedding_path) if not dqn_weights else torch.load(dqn_weights)
        self.dims = dims
        self.state = []
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon
        self.end_epsilon = end_epsilon
        self.rho = rho
        self.gamma = gamma
        self.max_moves = max_moves
        # remove when slot filler done
        self.action_vocab = action_vocab

    def explore(self):
        action = choice(self.action_vocab.num_to_word)
        print('exploring...', action)
        return action.split(' ')

    def exploit(self, text, output=True):
        rewards = {}
        for action in self.action_vocab:
            encoding = self.dqn.encode(text, action.split(' '))
            rewards[action] = self.dqn(encoding)
        max_action = max(rewards, key=rewards.get)
        if output:
            print('exploiting...', max_action)
        return max_action.split(' ')

    def callModel(self, text):
        action = self.explore() if random() < self.epsilon else self.exploit(text)
        print((self.init_epsilon - self.end_epsilon) /
              self.max_moves)
        self.epsilon -= (self.init_epsilon - self.end_epsilon) / \
            self.max_moves
        return action

    def loss_args(self, r, t: ReplayMemory):
        max_action = self.exploit(t.s_next, output=False)
        next_enc = self.dqn.encode(
            t.s_next, max_action)
        curr_enc = self.dqn.encode(t.s_t, t.a_t)
        max_r = torch.max(self.dqn(next_enc))
        y = r + (self.gamma * max_r)
        return y, self.dqn(curr_enc)[0]
