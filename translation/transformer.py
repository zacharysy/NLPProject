import torch
device = 'cpu'

import math, collections.abc, random, copy, sys, re

sys.path.append("./")

from training.layers import *

def progress(iterable):
        import os, sys
        if os.isatty(sys.stderr.fileno()):
            try:
                import tqdm
                return tqdm.tqdm(iterable)
            except ImportError:
                return iterable
        else:
            return iterable

class TranslationVocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""
    def __init__(self):
        words = {'<EOS>', '<UNK>'}
        self.num_to_word = list(words)
        self.word_to_num = {word:num for num, word in enumerate(self.num_to_word)}
    def add(self, word):
        if word in self: return
        num = len(self.num_to_word)
        self.num_to_word.append(word)
        self.word_to_num[word] = num
    def discard(self, word):
        raise NotImplementedError()
    def __contains__(self, word):
        return word in self.word_to_num
    def __len__(self):
        return len(self.num_to_word)
    def __iter__(self):
        return iter(self.num_to_word)

    def numberize(self, word):
        """Convert a word into a number."""
        if word in self.word_to_num:
            return self.word_to_num[word]
        else:
            return self.word_to_num['<UNK>']

    def denumberize(self, num):
        """Convert a number into a word."""
        return self.num_to_word[num]

def read_parallel(filename):
    """Read data from the file named by 'filename.'
    Argument: filename
    Returns: list of pairs of lists of strings. <EOS> is appended to all sentences.
    """

    lines = []
    data = []
    with open(filename) as file:
        for line in file:
            lines.append(line.strip())

    for i in range(0,len(lines)-1,2):
        fwords = list(map(lambda x: re.sub("\W", "", x).lower(), lines[i].strip().split())) + ["<EOS>"]
        ewords = lines[i+1].lower().strip().split() + ["<EOS>"]

        data.append((fwords, ewords))

    return data

def read_mono(filename):
    """Read sentences from the file named by 'filename.'

    Argument: filename
    Returns: list of lists of strings. <EOS> is appended to each sentence.
    """
    data = []
    for line in open(filename):
        words = line.split() + ['<EOS>']
        data.append(words)
    return data

class Encoder(torch.nn.Module):
    """Transformer encoder."""
    def __init__(self, vocab_size, dims):
        super().__init__()
        self.emb = Embedding(vocab_size, dims)
        self.pos = torch.nn.Parameter(torch.empty(1000, dims))
        torch.nn.init.normal_(self.pos, std=0.01)
        self.att1 = SelfAttention(dims)
        self.ffnn1 = TanhLayer(dims, dims, True)
        self.att2 = SelfAttention(dims)
        self.ffnn2 = TanhLayer(dims, dims, True)

    def forward(self, fnums):
        e = self.emb(fnums) + self.pos[:len(fnums)]
        h = self.att1(e)
        h = self.ffnn1(h)
        h = self.att2(h)
        h = self.ffnn2(h)
        return h

class Decoder(torch.nn.Module):
    """Transformer decoder."""

    def __init__(self, dims, vocab_size):
        super().__init__()
        self.emb = Embedding(vocab_size, dims)
        self.pos = torch.nn.Parameter(torch.empty(900, dims))
        torch.nn.init.normal_(self.pos, std=0.01)
        self.att = MaskedSelfAttention(dims)
        self.ffnn = TanhLayer(dims, dims, True)
        self.merge = TanhLayer(dims+dims, dims)
        self.out = SoftmaxLayer(dims, vocab_size)

    def start(self, fencs):
        """Return the initial state of the decoder.

        Since the only layer that has state is the attention, we just use
        its state. If there were more than one self-attention
        layer, this would be more complicated.
        """
        return (fencs, self.att.start())

    def input(self, state, enum):
        """Read in an English word (enum) and compute a new state from the old state (h)."""
        fencs, h = state
        flen = len(h)
        e = self.emb(enum) + self.pos[flen]
        h = self.att.input(h, e)
        return (fencs, h)

    def output(self, state):
        """Compute a probability distribution over the next English word."""
        fencs, h = state
        a = self.att.output(h)
        a = self.ffnn(a)
        c = attention(a, fencs, fencs)
        m = self.merge(torch.cat([c, a]))
        o = self.out(m)
        return o

class TranslationModel(torch.nn.Module):
    def __init__(self, fvocab, dims, evocab):
        super().__init__()

        # Store the vocabularies inside the Model object
        # so that they get loaded and saved with it.
        self.fvocab = fvocab
        self.evocab = evocab

        self.encoder = Encoder(len(fvocab), dims)
        self.decoder = Decoder(dims, len(evocab))

        # This is just so we know what device to create new tensors on
        self.dummy = torch.nn.Parameter(torch.empty(0))

    def logprob(self, fwords, ewords):
        """Return the log-probability of a sentence pair.

        Arguments:
            fwords: source sentence (list of str)
            ewords: target sentence (list of str)

        Return:
            log-probability of ewords given fwords (scalar)"""

        fnums = torch.tensor([self.fvocab.numberize(f) for f in fwords], device=self.dummy.device)
        fencs = self.encoder(fnums)
        state = self.decoder.start(fencs)
        logprob = 0.
        for eword in ewords:
            o = self.decoder.output(state)
            enum = self.evocab.numberize(eword)
            logprob += o[enum]
            state = self.decoder.input(state, enum)
        return logprob

    def translate(self, fwords):
        """Translate a sentence using greedy search.

        Arguments:
            fwords: source sentence (list of str)

        Return:
            ewords: target sentence (list of str)
        """
        numChoices = 10

        fnums = torch.tensor([self.fvocab.numberize(f) for f in fwords], device=self.dummy.device)
        fencs = self.encoder(fnums)
        state = self.decoder.start(fencs)
        ewords = []
        for i in range(100):
            o = self.decoder.output(state)

            topWord = self.evocab.denumberize(torch.argmax(o))
            if topWord == '<EOS>':
                break

            topk = torch.topk(o, numChoices)
            choice = random.randint(0,numChoices-1)
            enum = topk[1][choice].item()
            eword = self.evocab.denumberize(enum)
            if eword == '<EOS>': break
            ewords.append(eword)
            state = self.decoder.input(state, enum)

        return ewords

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='training data')
    parser.add_argument('infile', nargs='?', type=str, help='test data to translate')
    parser.add_argument('-o', '--outfile', type=str, help='write translations to file')
    parser.add_argument('--load', type=str, help='load model from file')
    parser.add_argument('--save', type=str, help='save model in file')
    args = parser.parse_args()

    if args.train:
        # Read training data and create vocabularies
        data = read_parallel(args.train)
        cutoff = int(len(data) * 0.7)

        traindata = data[:cutoff]
        devdata = data[cutoff:]


        fvocab = TranslationVocab()
        evocab = TranslationVocab()
        for fwords, ewords in traindata:
            fvocab |= fwords
            evocab |= ewords

        # Create model
        m = TranslationModel(fvocab, 64, evocab) # try increasing 64 to 128 or 256

    elif args.load:
        if args.save:
            print('error: --save can only be used with --train', file=sys.stderr)
            sys.exit()
        m = torch.load(args.load)

    else:
        print('error: either --train or --load is required', file=sys.stderr)
        sys.exit()

    if args.infile and not args.outfile:
        print('error: -o is required', file=sys.stderr)
        sys.exit()

    if args.train:
        opt = torch.optim.Adam(m.parameters(), lr=0.0003)

        best_dev_loss = None
        for epoch in range(10):
            random.shuffle(traindata)

            ### Update model on train

            train_loss = 0.
            train_ewords = 0
            for fwords, ewords in progress(traindata):
                loss = -m.logprob(fwords, ewords)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
                train_ewords += len(ewords)

            ### Validate on dev set and print out a few translations

            dev_loss = 0.
            dev_ewords = 0
            for line_num, (fwords, ewords) in enumerate(devdata):
                dev_loss -= m.logprob(fwords, ewords).item()
                dev_ewords += len(ewords)
                if line_num < 10:
                    print(f"{' '.join(fwords)}")
                    translation = m.translate(fwords)
                    print(' '.join(translation))
                    print()

            if best_dev_loss is None or dev_loss < best_dev_loss:
                best_model = copy.deepcopy(m)
                if args.save:
                    torch.save(m, args.save)
                best_dev_loss = dev_loss

            print(f'[{epoch+1}] train_loss={train_loss} train_ppl={math.exp(train_loss/train_ewords)} dev_ppl={math.exp(dev_loss/dev_ewords)}', flush=True)

        m = best_model

    ### Translate test set

    if args.infile:
        with open(args.outfile, 'w') as outfile:
            for fwords in read_mono(args.infile):
                translation = m.translate(fwords)
                print(' '.join(translation), file=outfile)
