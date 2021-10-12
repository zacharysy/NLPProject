"""
This file contains an implementation of a probability model which can
train on noun/verb pairs and learns a distribution P(verb | noun)
from that data.
"""

# Import libraries
from __future__ import annotations
import collections
import numpy as np


def sample_from_counter(counter: collections.Counter):
    """
    This function treats the counts in a `collections.Counter`
    object as counts in a probability distribution and samples
    from it.
    """

    # Split into values and counts
    values, counts = list(zip(*counter.items()))

    # Turn counts into a probability distribution
    distr = np.array(counts) / sum(counts)

    # Sample and return
    return np.random.choice(values, p=distr)


class NounVerb:
    """
    An adaptation of a general n-gram language model which models
    a conditional probability distribution between verbs and nouns.
    If a given noun is not present in the training data, probabilities
    are instead sampled from the marginalized P(verb) distribution.

    data: a list of (verb, noun) pairs to train on
    """

    def __init__(self, data: [(str, str)]):
        # Initialize counters
        self.marginal = collections.Counter()
        self.conditional = collections.defaultdict(collections.Counter)

        # Train the model
        for verb, noun in data:
            self.marginal[verb] += 1
            self.conditional[noun][verb] += 1

    def __call__(self, noun: str) -> str:
        """
        This function samples a verb from the distribution.
        """

        if noun in self.conditional:
            return sample_from_counter(self.conditional[noun])

        return sample_from_counter(self.marginal)
