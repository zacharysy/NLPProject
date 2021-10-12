#!/usr/bin/env bash

# Constants
VENVNAME="nlp_venv"

# Create virtual environment
python -m venv $VENVNAME
source $VENVNAME/bin/activate

# Install spaCy: https://spacy.io/usage
pip install -U pip setuptools wheel
pip install -U spacy[cuda112]         # modify for a given CUDA version
pip install -U textworld
python -m spacy download en_core_web_trf
