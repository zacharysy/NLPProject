# NLP Project

- **Note:** Run all commands from the root folder

## Requirements
- TextWorld requires non-python dependencies depending on the system
	- see more in the [TextWorld git repo][TW]

```
# macOS
brew install libffi curl git

# Debian/Ubuntu
sudo apt update && sudo apt install build-essential libffi-dev python3-dev curl git
```

- Install all the other dependencies using `setup.sh`

### Fix diaparser
- On line 113 of `/.nlp_venv/lib/python3.9/site-packages/diaparser/utils/transform.py` change to

```
def __getattr__(self, name):
	if name in self.__dict__:
		return self.__dict__[name]
	elif name in self.maps:
		return self.values[self.maps[name]]
	else:
		raise AttributeError()
```

## Benchmarks

### Generating Games
- To generate games used for the benchmark:
	- `python3 benchmark/benchmark.py --generate`
- The following parameters can be changed in `benchmark/benchmark.py` in the `__init__` of the `Benchmark` class
	- `games_root`: The folder where to store the generated games
	- `levels`: Number of levels to generate
	- `trials`: Number of trials to generate per level


### Running The Benchmark
- There are two benchmarks that can be run: Treasure or Zork
- There are two possible benchmark types to run: `treasure` and `zork`
	- `treasure` will run the treasure hunter benchmark from the [TextWorld paper][TW] for 30 levels, 50 trials per level, 1000 moves per trial, and records the average score and number of moves per level
	- `zork` runs Zork for 1000 moves and records the final score
- There are four agents corresponding to our four models: `baseline`, `rnn`, `transformer`, `salad`
- The benchmark can then be run with

```
python3 benchmark/benchmark.py --type <type> --agent <agent>
```

### Results
- The results of each of our models running the treasure benchmark can be found in `benchmark/results/treasure`
- The results of each of our models playing Zork can be found in `benchmark/results/zork`


## Baseline
- The baseline can be found in `agents.py` as `BaselineAgent`

### Outline
1. Gather a corpus of fiction novels stored in a text file where
   each line of the text file is one sentence from a novel
2. Use [spaCy](https://spacy.io/) to extract the main noun and
   verb from each sentence in the training data to create a
   reduced list of noun/verb pairs
3. Implement a lookup table which can transform a verb to its
   present-tense form and use it to convert all verbs from
   step 2 into the present tense
4. Create a probability distribution similar to a bigram language
   model which learns `P(verb | noun)` from the corpus created
   in the previous steps
5. Implement an agent using the
   [TextWorld][TW]
   library which iterates through the input for a given scenario,
   samples from `verb' = P(verb | noun)`, and attempts to submit
   `verb' noun` as a command for a given room
6. Test the agent on the
   [TextWorld][TW]
   benchmark.

Step Breakdown:

* Patrick Faley: 2 and 4
* Patrick Soga: 1 and 3
* Zachary Sy: 5 and 6

## Translation
- The RNN and Transformer models and training code can be found in `translation/`
- The text adventure agents can be found in `agents.py` as `RNNAgent` and `TransformerAgent`

## Student-created Automated Language-based Adventure Driver (SALAD)
-


[TW]: https://github.com/microsoft/textworld
