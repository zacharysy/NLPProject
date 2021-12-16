#!/usr/bin/env bash

# Constants
VENVNAME=".nlp_venv"

# Create virtual environment
python3 -m venv $VENVNAME
source $VENVNAME/bin/activate

# Install spaCy: https://spacy.io/usage
pip install -U pip setuptools wheel
pip install -r requirements.txt
# pip install -U spacy[cuda112]         # modify for a given CUDA version
# pip install -U textworld
# pip install -U tqdm
# pip install -U bs4
# pip install -U diaparser # Dependency Parser
# pip install -U pymagnitude # Word Embeddings helper
# pip install -U nltk
# pip install -U scikit-learn
# pip install -U stanford-openie

# Install extra stuff
python -m nltk.downloader all
# python -m spacy download en_core_web_trf
# python -m spacy download en

# note - pymagnitude seems to have issues with spacy, may need to uninstall

# Download a word embedding
curl http://magnitude.plasticity.ai/glove/medium/glove.6B.50d.magnitude > ./training/glove_weights.magnitude
