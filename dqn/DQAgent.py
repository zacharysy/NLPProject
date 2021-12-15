import abc
from random import choice, randint, random
import textworld
import torch

from knowledgeGraph.graph import KnowledgeGraph
from SALAD_bowl.integration import generate_actions
from .MA_DQN import MA_DQN
from .PA_DQN import PA_DQN
from .util import ReplayMemory
from pprint import pprint


def bad_feedback(feedback):
    # f = ' '.join(feedback)
    return "you don't" in feedback or "you can't" in feedback or "i don't" in feedback or \
        "i can't" in feedback  or "there is no" in feedback

class DQAgent(textworld.Agent, abc.ABC):
    @abc.abstractmethod
    def explore(self): pass

    @abc.abstractmethod
    def exploit(self): pass

    def act(self, game_state):
        game_state_str = ' '.join(game_state)
        while len(self.state) > 1 and self.state[-1] == game_state_str:
            self.state.pop()
        # if len(self.state) <= 1 and not bad_feedback(' '.join(game_state)):
        if not bad_feedback(game_state_str):
            self.state.append(game_state_str)
        return self.callModel(self.state[-1].split(' '))

    @ abc.abstractmethod
    def callModel(self): pass

    @ abc.abstractmethod
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
        self.epsilon -= (self.init_epsilon - self.end_epsilon) / self.max_moves
        return self.action_vocab.denumberize(action_num)

    def loss_args(self, r, t: ReplayMemory):
        action_num = self.action_vocab.numberize(t.a_t)
        next_enc = self.dqn.encode(t.s_next)
        curr_enc = self.dqn.encode(t.s_t)
        max_r = torch.max(self.dqn(next_enc))
        y = r + (self.gamma * max_r)
        return y, self.dqn(curr_enc)[action_num]


class PA_DQAgent(DQAgent):
    def __init__(self, dims=50, embedding_path=None, dqn_weights=None,
                 init_epsilon=1, end_epsilon=0.2, rho=0.25, gamma=0.5, transitions=1000,
                 slot_filler=None, knowledge_graph: KnowledgeGraph=None):
        self.dqn = PA_DQN(
            dims, embedding_path) if not dqn_weights else torch.load(dqn_weights)
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
        self.slot_filler = slot_filler
        self.knowledge_graph = knowledge_graph

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.dqn.state_dict())


    def explore(self, text):
        actions = generate_actions(text, self.knowledge_graph, self.slot_filler)
        action = choice(actions)
        return action.split(' ')

    def exploit(self, text, output=False, use_target_network=False):
        rewards = {}
        actions = generate_actions(text, self.knowledge_graph, self.slot_filler)

        for action in actions:
            if use_target_network:
                with torch.no_grad():
                    encoding = self.target_network.encode(
                        text, action.split(' ')).detach()
                    rewards[action] = self.target_network(encoding).detach()
            else:
                encoding = self.dqn.encode(text, action.split(' '))
                rewards[action] = self.dqn(encoding)

        max_action = max(rewards, key=rewards.get)
        if output:
            print('exploiting...', max_action)
        return max_action.split(' ')

    def callModel(self, text):
        action = self.explore(text) if random() < self.epsilon else self.exploit(text, output=False)
        self.epsilon -= (self.init_epsilon - self.end_epsilon) / \
            self.transitions
        return action

    def loss_args(self, r, t: ReplayMemory):
        max_action = self.exploit(
            t.s_next, output=False, use_target_network=True)
        with torch.no_grad():
            next_enc = self.target_network.encode(
                t.s_next, max_action).detach()
            max_r = torch.max(self.target_network(next_enc).detach())
            y = r + (self.gamma * max_r)

        curr_enc = self.dqn.encode(t.s_t, t.a_t)

        return y.detach(), self.dqn(curr_enc)[0]

        # max_action = self.exploit(t.s_next, output=False)
        # next_enc = self.dqn.encode(
        #     t.s_next, max_action)
        # curr_enc = self.dqn.encode(t.s_t, t.a_t)
        # max_r = torch.max(self.dqn(next_enc))
        # y = r + (self.gamma * max_r)
        # return y, self.dqn(curr_enc)[0]
