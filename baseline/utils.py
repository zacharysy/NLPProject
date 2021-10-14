"""
This file contains miscellaneous utility functions used in various
parts of the design process.
"""

from __future__ import annotations
import spacy


# Constants
SPACY_MODELNAME = 'en_core_web_trf'
VALID_VERB_POS = ['VERB']
VALID_NOUN_POS = ['NOUN', 'PROPN']
VALID_VERB_DEP = ['ROOT']
VALID_NOUN_DEP = ['dobj', 'pobj', 'npadvmod']

# Globals
SPACY_MODEL = None


def _initialize_spacy():
    """
    This function initializes the `SPACY_MODEL`
    variable using `SPACY_MODELNAME`
    """
    global SPACY_MODEL
    if SPACY_MODEL is None:
        SPACY_MODEL = spacy.load(SPACY_MODELNAME)


def extract_nounverb(sentence: str) -> (str, str):
    """
    This function takes in a sentence, performs part-of-speech
    analysis on the sentence, and extracts the relevant noun-verb
    pair. For example, consider the following sentence:

    The adventurer swung his sword.

    To convert this to a usable command, we would want to extract
    "swing sword". As a result, we are looking for the direct verb,
    which is "swung" in this example, as well as the a prepositional
    noun which acts as the object of the sentence.

    After the relevant words have been identified, the "lemma", which
    is the neutral form of the verb, is extracted.

    Input: an English sentence
    Output: verb, noun, or None if sentence not in a valid format
    """

    # Initilize spaCy parser
    _initialize_spacy()

    # Parse the sentence
    doc = SPACY_MODEL(sentence)

    # Find the relevant parts
    noun, verb = None, None

    for word in doc:
        # Check if we have the right verb
        if word.pos_ in VALID_VERB_POS and word.dep_ in VALID_VERB_DEP:
            verb = word.lemma_

        # Check if we have the right noun
        if word.pos_ in VALID_NOUN_POS and word.dep_ in VALID_NOUN_DEP:
            noun = word.lemma_

        # Return the discovered noun/verb
        if noun is not None and verb is not None:
            return (verb, noun)

    # Return that nothing has been found
    return None


if __name__ == "__main__":
    # Read in file
    with open('novels.txt', 'r') as f:
        lines = f.readlines()

    import tqdm

    # Parse the file
    out = list(filter(lambda x: x is not None, map(extract_nounverb, tqdm.tqdm(lines))))

    # Write the output
    with open('pairs.txt', 'w') as f:
        f.writelines([f'{v} {n}\n' for v, n in out])
