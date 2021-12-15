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
import random
import pymagnitude
import collections
import numpy as np
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer

import layers

import torch
import torch.nn as nn

# Constants
CSV_PATH = pathlib.Path('csv_data.csv')
TSV_PATH = pathlib.Path('tsv_data.csv')
EMBEDDING_PATH = pathlib.Path('glove_weights.magnitude')
MODEL_SAVE_PATH = pathlib.Path('sentence_classifier.ckpt')
RANDOM_SEED = 0
NUM_VERB_CLUSTERS = 20
NUM_PREP_CLUSTERS = 20
LEARNING_RATE = 0.0001
EPOCHS = 10


# Helper functions
def load_csv(path: pathlib.Path) -> [[str]]:
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

    # Make sure the tuples are of the appropriate length
    two_lines = list(filter(lambda x: len(x) == 2, two_lines))
    four_lines = list(filter(lambda x: len(x) == 2, four_lines))

    # Clean data
    data = two_lines + four_lines
    data = [[j.strip() for j in i] for i in data]

    return data


def load_tsv(path: pathlib.Path):
    """
    This function loads data in a more unified format from TSV files.
    """

    with open(path) as f:
        data = []

        for line in f.readlines():
            # First, strip the line
            line = line.strip()

            # Next, check if there are tabs
            line = line.split('\t' if '\t' in line else ' ')
            line = [i.strip() for i in line]

            if len(line) in [2, 4]:
                data.append(line)

    return data


def valid_sentence(tup: [str]) -> bool:
    """
    This function can be used to filter out any irrelevant tuples
    from the input.
    """

    multi_word_verb = len(tup[0].split()) > 1
    multi_word_preposition = len(tup) > 2 and len(tup[2].split()) > 1

    return not multi_word_verb and not multi_word_preposition

def create_sentence_standardizer():
    lemmatizer = WordNetLemmatizer()

    def standardize_sentence(tup: [str]) -> str:
        # Lemmatize verb
        tup[0] = lemmatizer.lemmatize(tup[0], 'v')

        # Split the others
        tup[1] = tup[1].split()

        if len(tup) > 2:
            tup[3] = tup[3].split()

        return tup

    return standardize_sentence

standardize_sentence = create_sentence_standardizer()


def create_clusterings(data: [[str]], embeddings, num_verb_clusters: int, num_prep_clusters: int, random_seed: int):
    """
    This function clusters the input verbs and prepositions based on their
    word embeddings.
    """

    # Extract the parts of speech
    verbs = [i[0] for i in data]
    nouns1 = [i[1] for i in data]
    preps = [i[2] for i in data if len(i) > 2]
    nouns2 = [i[3] for i in data if len(i) > 2]

    # Create clusters
    print('Creating verb clustering')
    verb_clustering = WordClustering(verbs, num_verb_clusters, embeddings, random_seed)
    print('Creating preposition clustering')
    prep_clustering = WordClustering(preps, num_prep_clusters, embeddings, random_seed)

    # Create vocab
    vocab = Vocab()
    for item in nouns1:
        vocab |= item
    for item in nouns2:
        vocab |= item

    return verb_clustering, prep_clustering, vocab


def handle_uncommon(data: [[str]]) -> [[str]]:
    """
    This function reads in all entries from `data`, finds which
    words only occur once, and replace them with UNK.

    Repurposed from Homework 4.
    """

    # Get uncommon words
    counts = collections.Counter()
    for line in data:
        if len(line) == 2:
            counts.update([line[0], *line[1]])

        else:
            counts.update([line[0], *line[1], line[2], *line[3]])
    uncommon = set([key for key, val in counts.items() if val == 1])

    # Replace uncommon words
    out = []
    for row in data:
        row_out = []
        for item in row:
            if type(item) is str:
                row_out.append(item if item not in uncommon else '<UNK>')

            else:
                row_out.append([i if i not in uncommon else '<UNK>' for i in item])

        out.append(row_out)

    return out


# Class definitions
class Vocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""
    def __init__(self):
        words = {'<EOS>', '<UNK>', '<NIL>', '<SEP>'}
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


class WordClustering:
    """
    This class combines a couple different functionalities. It takes
    a corpus as input and stores counts of each of the words. It also
    stores a class with each word, based on a scikit-learn clustering
    algorithm.
    """
    def __init__(self, words: [str], num_classes: int, embedding, random_seed: int):
        # Log hyperparameters
        self.num_classes = num_classes
        self.embedding = embedding

        # Create counts
        self.word_counts = collections.Counter(words)
        unique_words = list(self.word_counts.keys())

        # Cluster the verbs
        embeddings = self.embedding.query(unique_words)
        self.clustering = KMeans(n_clusters=num_classes, random_state=random_seed)
        class_assignments = self.clustering.fit_predict(embeddings)

        # Associate class assignments back with the words
        self.word_to_class = {word: assignment for word, assignment in zip(unique_words, class_assignments)}
        self.class_to_word = {num: [(i, self.word_counts[i]) for i,j in zip(unique_words, class_assignments) if j == num] for num in range(self.num_classes)}
        self.class_to_word = {key: sorted(val, key=lambda x: -1*x[1]) for key, val in self.class_to_word.items()}

    def embed_word(self, word: str) -> int:
        # Always generates an embedding
        word_embedding = self.embedding.query(word).reshape(1, -1).astype(np.float32)
        word_clustering = self.clustering.predict(word_embedding)

        return int(word_clustering)

    def classify_word(self, word: str) -> int:
        return self.word_to_class.get(word, self.embed_word(word))

    def sample(self, class_num: int, mode: str = 'top'):
        """
        Draws a sample from a given class. Several different sampling methods
        can be performed.

        top - take the most frequent word from that class
        uniform - draw uniformly from the class
        sample - randomly sample from the distribution generated by the word counts
        top5 - draw uniformly from the top 5 most common
        """

        # Get relevant list of words
        cluster_words = self.class_to_word[class_num]

        # Get the top
        if mode == 'top':
            return cluster_words[0][0]

        if mode == 'uniform':
            return random.choice(cluster_words)[0]

        if mode == 'sample':
            words, counts = list(zip(*cluster_words))
            return np.random.choice(words, p=np.array(counts)/sum(counts))

        return random.choice(cluster_words[:5])[0]


class AssignmentClassifier(nn.Module):
    """
    This class implements a neural network for dialogue act classification.
    It has the following shape:

    Input -> Embedding -> Self Attention -> Linear -> Reduce -> Softmax

    The number of self-attention and linear layers are tunable parameters.

    Methods:
    forward :: noun(s) -> predicted class label
    get_class_name :: verb, maybe prep -> actual class label
    run_models :: sentence -> predicted, actual class labels
    get_class_words :: class label -> verb, maybe prep
    get_full_sentence :: noun(s) -> full sentence
    """
    def __init__(self,
                 word_vocab: Vocab,
                 attention_size: int,
                 num_attention_layers: int,
                 linear_size: int,
                 num_linear_layers: int,
                 verb_clusters: WordClustering,
                 prep_clusters: WordClustering):

        super().__init__()

        # Save parameters
        self.word_vocab = word_vocab
        self.num_verb_clusters = verb_clusters.num_classes
        self.num_prep_clusters = prep_clusters.num_classes
        self.num_classes = self.num_verb_clusters * (self.num_prep_clusters + 1)
        self.attention_size = attention_size
        self.num_attention_layers = num_attention_layers
        self.linear_size = linear_size
        self.num_linear_layers = num_linear_layers
        self.verb_clusters = verb_clusters
        self.prep_clusters = prep_clusters

        # Initialize layers
        self.embedding = layers.Embedding(len(word_vocab), attention_size)
        self.attention = [layers.SelfAttention(attention_size) for _ in range(num_attention_layers)]

        if num_linear_layers == 0:
            self.linear_layers = []
            self.softmax = layers.SoftmaxLayer(attention_size, self.num_classes)

        else:
            self.linear_layers = [layers.LinearLayer(attention_size, linear_size)] + [layers.LinearLayer(linear_size, linear_size) for _ in range(num_linear_layers-1)]
            self.softmax = layers.SoftmaxLayer(linear_size, self.num_classes)

    def individual_to_whole(self, verb_clustering: int, prep_clustering: int) -> int:
        return verb_clustering + (prep_clustering+1) * self.num_verb_clusters

    def whole_to_individual(self, clustering: int) -> (int, int):
        verb_clustering = clustering % self.num_verb_clusters
        prep_clustering = (clustering // self.num_verb_clusters) - 1

        return verb_clustering, prep_clustering

    def forward(self, in_words: [str]) -> torch.Tensor:
        # Step 1: convert words to indices
        word_encoding = torch.tensor([self.word_vocab.numberize(word) for word in in_words])

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

    def get_class_name(self, verb: str, prep: str = None) -> int:
        """
        This method goes from a verb and (optionally) preposition to the
        corresponding class label
        """

        # Compute clusterings
        verb_clustering = self.verb_clusters.classify_word(verb)
        prep_clustering = self.prep_clusters.classify_word(prep) if prep is not None else -1

        # Compute the class label
        class_label = self.individual_to_whole(verb_clustering, prep_clustering)

        return int(class_label)

    def run_models(self, in_words: [str]) -> torch.Tensor:
        """
        This method is a convenience method which allows you to train on the
        full tuple. You get both the result of a forward call, as well as the
        class name.
        """

        # Parse out
        if len(in_words) == 2:
            verb, noun = in_words
            prediction = self.forward(noun)
            ground_truth = self.get_class_name(verb)

        else:
            verb, noun1, prep, noun2 = in_words
            prediction = self.forward(noun1 + ['<SEP>'] + noun2)
            ground_truth = self.get_class_name(verb, prep)

        return prediction, ground_truth

    def get_class_words(self, label: int, mode: str = 'top') -> [str]:
        """
        See the WordClustering class methods for the `mode` uses
        """
        # Handle bad input
        if label < 0 or label >= self.num_classes:
            return ['<NIL>']

        # Handle prep-less input
        if label < self.num_verb_clusters:
            return [self.verb_clusters.sample(label, mode=mode)]

        # Handle the preposition and verb
        verb_label, prep_label = self.whole_to_individual(label)
        verb_choice = self.verb_clusters.sample(verb_label, mode=mode)
        prep_choice = self.prep_clusters.sample(prep_label, mode=mode)

        return verb_choice, prep_choice

    def get_full_sentence(self, in_words: [str], mode: str = 'top') -> [str]:
        # Get classification results
        class_probs = self.forward(in_words)
        best_class = torch.argmax(class_probs).item()

        # Get corresponding verbs
        class_words = self.get_class_words(best_class, mode)

        if len(class_words) == 2 and len(in_words) == 2:
            return [class_words[0], in_words[0], class_words[1], in_words[1]]

        return [class_words[0], in_words[0]]


def load_model(csv_path: pathlib.Path,
               tsv_path: pathlib.Path,
               embedding_path: pathlib.Path,
               num_verb_clusters: int,
               num_prep_clusters: int,
               random_seed: int = 0,
               weight_path: pathlib.Path = None):

    # Load the data
    print('Loading data from CSV/TSV')
    data = load_csv(csv_path) + load_tsv(tsv_path)

    # Preprocess the data
    print('Ensuring we have valid sentences')
    data = list(map(standardize_sentence, filter(valid_sentence, data)))
    data = handle_uncommon(data)

    # Load embeddings
    embeddings = pymagnitude.Magnitude(embedding_path)

    # Perform k-means clustering
    verb_clusters, prep_clusters, vocab = create_clusterings(data,
                                                             embeddings,
                                                             num_verb_clusters,
                                                             num_prep_clusters,
                                                             random_seed)

    # Create the model
    model = AssignmentClassifier(vocab,
                                 200, 1, 200, 0,
                                 verb_clusters,
                                 prep_clusters)

    # Load model weights
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))

    return model, data


if __name__ == '__main__':
    # Load the model
    model, data = load_model(CSV_PATH,
                             TSV_PATH,
                             EMBEDDING_PATH,
                             NUM_VERB_CLUSTERS,
                             NUM_PREP_CLUSTERS,
                             RANDOM_SEED,
                             None)

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
        for i, item in tqdm.tqdm(list(enumerate(data))):
            # Forward pass
            prediction, ground_truth = model.run_models(item)
            prediction = prediction.unsqueeze(0)

            # Prepare the output
            target = torch.tensor([ground_truth], dtype=torch.int64)

            # Calculate loss
            loss_val = loss_fn(prediction, target)

            # Perform gradient stuff
            optim.zero_grad()
            loss_val.backward()
            optim.step()

            # Add running
            running_loss += loss_val.detach().item()

            if i % 10000 == 0:
                # Save model weights
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # Report train loss/dev F1
        print()
        print(f'Train Loss: {running_loss/len(data):.4f}')

        # Save final model weights
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
