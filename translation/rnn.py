import collections
import random
import re
import torch
import datetime
import argparse


def progress(iterable):
    import os
    import sys
    if os.isatty(sys.stderr.fileno()):
        try:
            import tqdm
            return tqdm.tqdm(iterable)
        except ImportError:
            return iterable
    else:
        return iterable


def read_train_data(filename):
    input_vocab = []
    output_responses = []
    lines = []
    with open(filename) as file:
        for line in file:
            lines.append(line.strip())
    for i, line in enumerate(lines):
        if i % 2 == 0:
            input_vocab.append({'line': line.strip().split(
                ' ') + ['<EOS>'], 'command': lines[i+1]})
        else:
            output_responses.append([line.strip()])
    return input_vocab, output_responses


class Vocab:
    def __init__(self, counts, size):
        if not isinstance(counts, collections.Counter):
            raise TypeError('counts must be a collections.Counter')
        words = {'<EOS>', '<UNK>'}
        for word, _ in counts.most_common():
            words.add(word)
            if len(words) == size:
                break
        self.num_to_word = list(words)
        self.word_to_num = {word: num for num,
                            word in enumerate(self.num_to_word)}

    def __len__(self):
        return len(self.num_to_word)

    def __iter__(self):
        return iter(self.num_to_word)

    def numberize(self, word):
        if word in self.word_to_num:
            return self.word_to_num[word]
        else:
            return self.word_to_num['<UNK>']

    def denumberize(self, num):
        return self.num_to_word[num]


class Model(torch.nn.Module):
    def __init__(self, vocab: Vocab, response_vocab: Vocab, d: int, device: str):
        super().__init__()
        self.vocab = vocab
        self.response_vocab = response_vocab
        self.d = d
        self.device = device
        self.W = torch.nn.Parameter(torch.normal(
            mean=0.0, std=0.01, size=[d, d, d], requires_grad=True, device=device))
        self.B = torch.nn.Parameter(torch.normal(mean=0.0, std=0.01, size=[
                                    d, len(vocab)], requires_grad=True, device=device))
        self.c = torch.nn.Parameter(torch.normal(
            mean=0.0, std=0.01, size=[d], requires_grad=True, device=device))
        self.D = torch.nn.Parameter(torch.normal(mean=0.0, std=0.01, size=[
                                    len(vocab), d], requires_grad=True, device=device))
        self.e = torch.nn.Parameter(torch.normal(
            mean=0.0, std=0.01, size=[len(vocab)], requires_grad=True, device=device))
        self.h_initial = torch.nn.Parameter(torch.normal(
            mean=0.0, std=0.01, size=[d], requires_grad=True, device=device))
        self.prev_h = self.h_initial.clone()
        self.linear = torch.nn.Linear(
            in_features=len(vocab), out_features=len(response_vocab), device=device)

    def generate_state(self, one_hot_idx):
        v = self.B[:, one_hot_idx]
        z = torch.einsum('ijk,j,k->i', self.W, self.prev_h, v)
        return torch.tanh(z + self.c)

    def forward(self, q):
        return torch.log_softmax(self.linear(torch.flatten((self.D @ q + self.e))), dim=0)

    def start(self):
        return self.h_initial

    def read(self, q, a):
        one_hot_idx = self.vocab.numberize(a)
        self.prev_h = q.clone()
        return self.generate_state(one_hot_idx)

    def best(self, q):
        predicted_dist = self.forward(q)
        char = self.response_vocab.denumberize(predicted_dist.argmax())
        return char

    def restart(self):
        self.prev_h = self.h_initial.clone()

    def train(self, data=None,  epochs=30, optimizer=None):
        # if not data or not optimizer:
        #     print('Missing training data or optimizer')
        #     return
        # from tutorial notebook
        for epoch in range(epochs):
            random.shuffle(data)
            cutoff = int(len(data) * 0.8)
            dev_data = data[cutoff:]

            train_loss = 0
            for line in progress(data[:cutoff]):
                self.restart()
                states = [self.start()]
                loss = 0.
                for word in line['line']:
                    q = states[-1]

                    predicted_dist = self.forward(q)
                    loss -= predicted_dist[predicted_dist.argmax()]
                    q = self.read(q, word)
                    states.append(q)

                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                optimizer.step()

            self.eval()
            correct_commands = 0
            incorrect_commands = 0

            for line in dev_data:
                self.restart()
                states = [self.start()]
                for word in line['line']:
                    q = states[-1]

                    predicted_dist = self.forward(q)
                    if line['command'] == self.response_vocab.denumberize(predicted_dist.argmax()):
                        correct_commands += 1
                    else:
                        incorrect_commands += 1
                    q = self.read(q, word)
                    states.append(q)

            print(f'[{epoch+1}] train_loss={train_loss}', flush=True)
            print(
                f'[{epoch+1}] dev correct/incorrect={correct_commands}/{incorrect_commands}', flush=True)

        now = datetime.datetime.now()
        filename = f'epoch_{epoch}-{now.strftime("%d-%m_%H:%M")}'
        torch.save(self, filename)


def preprocess_line(text):
    cleaned = re.sub('([".,!?()])', r' \1 ', text)
    line = cleaned.split()
    return line


def predict(model, text):
    # model.eval()
    states = [model.start()]
    q = states[-1]
    for word in text:
        word = model.best(q)
        q = model.read(q, word)
        states.append(q)
    return word


def main(args):
    if torch.cuda.device_count() > 0:
        print(f'Using GPU ({torch.cuda.get_device_name(0)})')
        device = 'cuda'
    else:
        print('Using CPU')
    device = 'cpu'

    input_train, output_responses = read_train_data(args.train)

    words = collections.Counter()
    for line in input_train:
        words.update(line['line'])
    vocab = Vocab(words, 750)

    commands = collections.Counter()
    for response in output_responses:
        commands.update(response)
    response_vocab = Vocab(commands, 300)

    model = Model(vocab, response_vocab, 64, device)
    o = torch.optim.SGD(model.parameters(), lr=0.1)
    model.train(data=input_train, epochs=10, optimizer=o)
    model.save(args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', type=str)
    parser.add_argument('--save-path', dest='save', type=str)
    args = parser.parse_args()
    main(args)
