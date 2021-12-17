from __future__ import annotations
import abc
import sys
from SALAD_bowl.integration import generate_actions

from dqn.PA_DQN import PA_DQN
from dqn.util import ReplayMemory, bad_feedback
sys.path.append("./")

from baseline.nounverb import NounVerb
import random
import re
import spacy
from pprint import pprint
import translation.rnn as rnn
from translation.transformer import TranslationVocab, Encoder, Decoder, TranslationModel
from random import choice, random
import torch

import textworld


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
        data = list(filter(lambda x: len(x) == 2, map(
            lambda x: x.strip().split(' '), data)))

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

    def __init__(self, weight_path='./translation/transformer'):
        # Load model
        # self.model = transformer.load_model(weight_path)
        self.model = torch.load(weight_path)
        self.state = []

    def act(self, game_state, reward, done):
        self.state.append(game_state)
        while self.state[-1].feedback == game_state.feedback and len(self.state) > 1:
            self.state.pop()

        text = re.sub(
            "(\W|_)+", " ", self.state[-1].feedback).strip().lower().split() + ["<EOS>"]

        return self.callModel(text)

    def callModel(self, text):
        # Run the prediction
        prediction = self.model.translate(text)

        return ' '.join(prediction)


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


class DQAgent(textworld.Agent, abc.ABC):
    @abc.abstractmethod
    def explore(self): pass

    @abc.abstractmethod
    def exploit(self): pass

    def act(self, game_state, reward=None, done=False):
        game_state_str = game_state['feedback']
        while len(self.state) > 1 and self.state[-1] == game_state_str:
            self.state.pop()
        if not bad_feedback(game_state_str):
            self.state.append(game_state_str)

        return self.callModel(self.state[-1].split(' '))

    @ abc.abstractmethod
    def callModel(self): pass

    @ abc.abstractmethod
    def loss_args(self): pass

    def save(self, filename):
        return torch.save(self.dqn, filename)


class PA_DQAgent(DQAgent):
    def __init__(self, dims=50, embedding_path=None, dqn_weights=None,
                 init_epsilon=1, end_epsilon=0.2, rho=0.25, gamma=0.5, transitions=1000,
                 slot_filler=None, knowledge_graph=None, # slot_filler_weights=None,
                 should_train=True):
        self.dqn = PA_DQN(
                    dims, embedding_path)
        if dqn_weights:
            self.dqn.load_state_dict(torch.load(dqn_weights))
        self.target_network = PA_DQN(dims, embedding_path)
        self.target_network.eval()
        self.dims = dims
        self.state = []
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon
        self.end_epsilon = end_epsilon
        self.rho = rho
        self.gamma = gamma
        self.transitions = transitions
        self.slot_filler = slot_filler # if not slot_filler_weights else torch.load(slot_filler_weights)
        self.knowledge_graph = knowledge_graph
        self.should_train = should_train

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.dqn.state_dict())

    def reset_graph(self):
        self.knowledge_graph.flush()

    def explore(self, text):
        # print('Entering `explore`')
        actions = generate_actions(text, self.knowledge_graph, self.slot_filler)
        action = choice(actions)
        # print('Exiting `explore`')
        return action.split(' ')

    def filter_action(self, action):
        is_list = type(action) is list

        if not is_list:
            action = ' '.join(action.split('\n'))
            action = action.split(' ')

        if len(action) > 150:
            action = action[:75] + action[-75:]

        if not is_list:
            action =' '.join(action)

        return action

    def exploit(self, text, output=False, use_target_network=False):
        # print('Entering `exploit`')
        rewards = {}
        actions = generate_actions(text, self.knowledge_graph, self.slot_filler)

        for action in actions:
            if use_target_network:
                with torch.no_grad():
                    encoding = self.target_network.encode(
                        text, self.filter_action(action).split(' ')).detach()
                        # text, action.split(' ')).detach()
                    rewards[action] = self.target_network(encoding).detach()
            else:
                encoding = self.dqn.encode(self.filter_action(text), self.filter_action(action).split(' ')) #.split(' '))
                rewards[action] = self.dqn(encoding)

        max_action = max(rewards, key=rewards.get)
        if output:
            print('exploiting...', max_action)
        # print('Exiting `exploit`')
        return max_action.split(' ')

    def callModel(self, text):
        # Clean input text
        text = [i.strip() for i in text if len(i.strip()) > 0 and '$$' not in i]

        action = self.explore(text) if random() < self.epsilon else self.exploit(text, output=False)
        if self.should_train:
            self.epsilon -= (self.init_epsilon - self.end_epsilon) / \
                self.transitions
        # print(text)
        # print(f'Call Input: {text}')
        # print(f'Action: {action}')
        # print()

        return action

    def loss_args(self, r, t: ReplayMemory):
        max_action = self.exploit(
            t.s_next, output=False, use_target_network=self.should_train)
        with torch.no_grad():
            next_enc = self.target_network.encode(
                t.s_next, max_action).detach()
            max_r = torch.max(self.target_network(next_enc).detach())
            y = r + (self.gamma * max_r)

        curr_enc = self.dqn.encode(t.s_t, t.a_t)

        return y.detach(), self.dqn(curr_enc)[0]


if __name__ == "__main__":
    pass
