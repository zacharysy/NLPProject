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

## Translation
- The RNN and Transformer models and training code can be found in `translation/`
- The text adventure agents can be found in `agents.py` as `RNNAgent` and `TransformerAgent`

## Student-created Automated Language-based Adventure Driver (SALAD)

- Knowledge Graph Representation: The code to create and parse the knowledge graph,
  as well as an example of how to use `OpenIE`, can be found in `knowledgeGraph/`.
  Specifically, `graph.py` implements the logic actually used in SALAD.
- Heuristic-Based Slot Filler: The code for the heuristic-based slot filler can be
  found in `heuristicSlotFilling/`. The logic is implemented in `classifier.py`,
  with data from `target_attribute_scores.csv` and `templates.txt`.
- Learning-Based Slot Filler: The code for the learning-based slot filler can be
  found in `traiining/`. The `templating.py` file there contains the main logic.
- Deep Q Network: The actual reinforcement learning code can be found in `dqn/`.
  It has different files containing the agents, as well as some helper functions.
- Full System: The full system can be found in the `SALAD_bowl/` directory.
  The `salad.py` file implements training of the deep Q agent, while `integration.py`
  contains code for integrating all of the above parts.
- The text adventure agent can be found in `agents.py` as `salad-h` and `salad-l`,
  where `salad-h` uses the heuristic-based slot filler and `salad-l` uses the
  learning-based slot filler.


[TW]: https://github.com/microsoft/textworld
