#!/usr/bin/env bash

# Constants
VENVNAME=".nlp_venv"

# Create virtual environment
python3 -m venv $VENVNAME
source $VENVNAME/bin/activate

# Install spaCy: https://spacy.io/usage
pip install -U pip setuptools wheel
pip install -U spacy[cuda112]         # modify for a given CUDA version
pip install -U textworld
pip install -U tqdm
pip install -U bs4
pip install -U diaparser # Dependency Parser
pip install -U magnitude # Word Embeddings helper
python -m spacy download en_core_web_trf
