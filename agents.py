import translation.rnn as rnn
from translation.transformer import TranslationVocab, Encoder, Decoder, TranslationModel
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


class DQAgent(textworld.Agent):
    def __init__(self, word_vocab, action_vocab, dims, dqn_weights=None, *args, **kwargs):
        self.dqn = MA_DQN(word_vocab, action_vocab,
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
