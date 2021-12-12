"""
This file contains the logic for reducing the input tuples to a set of
clusters. Tuples can take a few different forms. The first is:

(verb, noun)

The second is:

(verb, noun1, preposition, noun2)
"""

# Import libraries
import csv
import tqdm
import pathlib
import pymagnitude
from sklearn.cluster import KMeans

import layers

import torch
import torch.nn as nn

# Constants
DATA_PATH = pathlib.Path('depTuples.csv')
EMBEDDING_PATH = pathlib.Path('glove_weights.magnitude')
MODEL_SAVE_PATH = pathlib.Path('sentence_classifier.ckpt')
NUM_VERB_CLUSTERS = 20
NUM_PREP_CLUSTERS = 20
LEARNING_RATE = 0.0001
EPOCHS = 10

def load_data(path: pathlib.Path):
    """
    This function loads the data into a list of tuples.
    """

    # Only works if we have TSV
    # return [[i.strip() for i in row] for row in csv.reader(open(path))]

    with open(path) as f:
        lines = f.readlines()
        lines = [i.strip() for i in lines]

    # Get tuples by length
    two_lines = [i[:-2].split(', ') for i in lines if i[-2:] == ',,']
    four_lines = [i.split(', ') for i in lines if i[-2:] != ',,']

    # Clean data
    data = two_lines + four_lines
    data = [[j.strip() for j in i] for i in data]

    return data


def valid_sentence(tup: [str]) -> bool:
    """
    This function can be used to filter out any irrelevant tuples
    from the input.
    """

    multi_word_verb = len(tup[0].split()) > 1
    multi_word_preposition = len(tup) > 2 and len(tup[2].split()) > 1

    return not multi_word_verb and not multi_word_preposition


def create_clusterings(data: [[str]], embeddings, num_verb_clusters: int, num_prep_clusters: int):
    """
    This function clusters the input verbs and prepositions based on their
    word embeddings.
    """

    # Extract the verbs
    verbs = [i[0] for i in data]
    preps = [i[2] for i in data if len(i) > 2]

    # Get the encodings
    print('Encoding verbs')
    verb_encodings = embeddings.query(verbs)
    print('Encoding prepositions')
    prep_encodings = embeddings.query(preps)

    # Initialize the models
    verb_clusters = KMeans(n_clusters=num_verb_clusters)
    prep_clusters = KMeans(n_clusters=num_prep_clusters)

    # Fit the models
    print('Training verb clusters')
    verb_clusters.fit(verb_encodings)
    print('Training preposition clusters')
    prep_clusters.fit(prep_encodings)

    return verb_clusters, prep_clusters


def handle_uncommon(data: [[str]]) -> [[str]]:
    """
    This function reads in all entries from `data`, finds which
    words only occur once, and replace them with UNK.

    Repurposed from Homework 4.
    """

    # Get uncommon words
    counts = collections.Counter([j for i in data for j in i])
    uncommon = set([key for key, val in counts.items() if val == 1])

    # Replace uncommon words
    out = [[j if j not in uncommon else '<UNK>' for j in i] for i in data]

    return out


class Vocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""
    def __init__(self):
        words = {'<EOS>', '<UNK>', '<NIL>'}
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


class AssignmentClassifier(nn.Module):
    """
    This class implements a neural network for dialogue act classification.
    It has the following shape:

    Input -> Embedding -> Self Attention -> Linear -> Reduce -> Softmax

    The number of self-attention and linear layers are tunable parameters.
    """
    def __init__(self,
                 word_vocab: Vocab,
                 num_verb_clusters: int,
                 num_prep_clusters: int,
                 attention_size: int,
                 num_attention_layers: int,
                 linear_size: int,
                 num_linear_layers: int,
                 verb_clusters,
                 prep_clusters,
                 word_embeddings):

        super().__init__()

        # Save parameters
        self.word_vocab = word_vocab
        self.num_verb_clusters = num_verb_clusters
        self.num_prep_clusters = num_prep_clusters
        self.num_classes = num_verb_clusters * (NUM_PREP_CLUSTERS + 1)
        self.attention_size = attention_size
        self.num_attention_layers = num_attention_layers
        self.linear_size = linear_size
        self.num_linear_layers = num_linear_layers
        self.verb_clusters = verb_clusters
        self.prep_clusters = prep_clusters
        self.word_embeddings = word_embeddings

        # Initialize layers
        self.embedding = layers.Embedding(len(word_vocab), attention_size)
        self.attention = [layers.SelfAttention(attention_size) for _ in range(num_attention_layers)]

        if num_linear_layers == 0:
            self.linear_layers = []
            self.softmax = layers.SoftmaxLayer(attention_size, self.num_classes)

        else:
            self.linear_layers = [layers.LinearLayer(attention_size, linear_size)] + [layers.LinearLayer(linear_size, linear_size) for _ in range(num_linear_layers-1)]
            self.softmax = layers.SoftmaxLayer(linear_size, self.num_classes)

    def forward(self, in_sentence: [str]) -> torch.Tensor:
        # Step 1: convert words to indices
        word_encoding = torch.tensor([self.word_vocab.numberize(word) for word in in_sentence])

        # Step 2: compute the embedding of the encoding
        emb = self.embedding(word_encoding)

        # Step 3: self-attention layers
        attention_out = emb
        for layer in self.attention:
            attention_out = layer(attention_out)

        # Step 4: linear layers
        linear_out = attention_out
        for layer in self.linear_layers:
            linear_out = layer(linear_out)

        # Step 5: reduce
        reduced_out = torch.sum(linear_out, axis=0)

        # Feed through softmax
        out = self.softmax(reduced_out)

        return out

    # def predict_tuple(self, item: [str]) -> torch.Tensor:
    #     # Generate labels and such
    #     if len(item) == 2:
    #         verb, noun = item
    #         verb_class = self.verb_clusters.predict(self.embeddings.query(verb).reshape(1, -1))
    #         inp = noun.split()
    #         overall_class = verb_class

    #     else:
    #         verb, noun1, prep, noun2 = item
    #         verb_class = self.verb_clusters.predict(self.embeddings.query(verb).reshape(1, -1))
    #         prep_class = self.prep_clusters.predict(self.embeddings.query(prep).reshape(1, -1))
    #         inp = noun1.split() + noun2.split()
    #         overall_class = verb_class + (prep_class + 1) * NUM_VERB_CLUSTERS

    #     return self.forward()


if __name__ == '__main__':
    # Load the data
    data = load_data(DATA_PATH)

    # Preprocess the data
    data = list(filter(valid_sentence, data))

    # Load embeddings
    embeddings = pymagnitude.Magnitude(EMBEDDING_PATH)

    # Perform k-means clustering
    verb_clusters, prep_clusters = create_clusterings(data, embeddings, NUM_VERB_CLUSTERS, NUM_PREP_CLUSTERS)


    # Create the vocab
    vocab = Vocab()
    for line in data:
        vocab |= {line[1]}
        if len(line) > 2:
            vocab |= {line[3]}

    # Create the model
    model = AssignmentClassifier(vocab, num_labels, 200, 1, 200, 1)

    # Initialize optimizer and loss function
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # Main loop
    for epoch in range(EPOCHS):
        print(f'--- Epoch {epoch+1}/{EPOCHS} ---')

        # Initialize trackers
        running_loss = 0
        random.shuffle(data)

        # Iterate through training data
        for item in tqdm.tqdm(data):

            # Forward pass
            out = model(inp).unsqueeze(0)

            # Prepare the output
            target = torch.tensor([overall_class], dtype=torch.int64)

            # Calculate loss
            loss_val = loss_fn(out, target)

            # Perform gradient stuff
            optim.zero_grad()
            loss_val.backward()
            optim.step()

            # Add running
            running_loss += loss_val.detach().item()

        # Report train loss/dev F1
        print()
        print(f'Train Loss: {running_loss/len(train_words):.4f}')

        # Save model weights
        torch.save(model, SAVE_PATH)
