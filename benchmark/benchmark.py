from __future__ import annotations
import sys
sys.path.append("./")

from agents import RNNAgent, BaselineAgent, TransformerAgent
from translation.transformer import TranslationVocab, Encoder, Decoder, TranslationModel
import argparse
from collections import defaultdict
from pprint import pprint
import json
import os
import tqdm

import textworld
from textworld.generator.game import GameOptions
from textworld.generator import compile_game
from translation.rnn import *
# from baseline.ExampleAgent import CustomAgent


class Benchmark:
    def __init__(self):
        self.games_root = "./benchmark/games"
        self.levels = 30
        self.trials = 50
        self.max_moves = 1000
        self.levels_step = 1
        self.save_to = None

    # Create a single treasure_hunter game with the given level and saves it to self.games_root
    def createGame(self, level: int, trial: int) -> str:
        options = GameOptions()
        options.path = f"{self.games_root}/{level}/{trial}/"

        game = textworld.challenges.treasure_hunter.make(
            settings={"level": level}, options=options)

        path = compile_game(game, options=options)

        return path

    # Creates trial number of games for each level
    def generateGames(self,
                      levels: Optional(int) = None,
                      trials: Optional(int) = None
                      ):
        levels = levels or self.levels
        trials = trials or self.trials

        for i in range(levels):
            level = i+1

            for j in range(trials):
                trial = j + 1

                game = self.createGame(level, trial)

                print(f"Level {level} Trial {trial} created", end="\r")

    # Create a JSON file to list out all the available games in the given folder path
    def registerGames(self, folder_path: Optional(str) = None):
        folder_path = folder_path or self.games_root

        '''
        {
            level : {
                trial: path
            }
        }
        '''
        game_directory = defaultdict(dict)

        with os.scandir(folder_path) as levels:
            for level in filter(lambda x: os.path.isdir(x), levels):
                with os.scandir(level) as trials:
                    for trial in filter(lambda x: os.path.isdir(x), trials):
                        with os.scandir(trial) as games:
                            game = list(
                                filter(lambda x: x.path.endswith(".ulx"), games))[0]
                            game_directory[level.name][trial.name] = game.path

        with open(os.path.join(folder_path, "games.json"), "w") as file:
            file.write(json.dumps(game_directory))

    # Run the benchmark up to the given `levels` and `trials` number of trials with at most `max_moves` number of moves per game
    def runBenchmark(self,
                     agent: textworld.Agent,
                     directory_path: Optional(str) = None,
                     levels: Optional(int) = None,
                     step: Optional(int) = None,
                     trials: Optional(int) = None,
                     max_moves: Optional(int) = None
                     ):
        directory_path = directory_path or self.games_root
        levels = levels or self.levels
        step = step or self.levels_step
        trials = trials or self.trials
        max_moves = max_moves or self.max_moves
        games = None

        '''
        {
            level : {
                averageScore:
                averageMoves:
            }
        }
        '''
        results = {}

        with open(os.path.join(directory_path, "games.json"), "r") as file:
            games = json.loads(file.read())

        if games is None:
            return

        for level in range(1, levels + 1, step):
            print(f"Level {level}")
            results[level] = {
                "average_score": 0,
                "average_moves": 0
            }

            for trial in tqdm.tqdm(range(1, trials + 1)):
                score, moves = self.runGame(
                    agent, games[str(level)][str(trial)], max_moves)
                results[level]["average_score"] += score
                results[level]["average_moves"] += moves

            results[level]["average_score"] /= trials
            results[level]["average_moves"] /= trials

            print(
                f"average score: {results[level]['average_score']}\t average moves: {results[level]['average_moves']}\n")

        print()
        self.saveResults(results)
        pprint(results)

    # run a single game
    def runGame(self, agent: textworld.Agent, game_path: str, max_moves: int) -> (int, int):
        env = textworld.start(game_path)
        game_state = env.reset()
        reward, done, moves = 0, False, 0

        while not done and moves < max_moves:
            command = agent.act(game_state, reward, done)
            game_state, reward, done = env.step(command)
            moves += 1

        score = game_state.score

        # If it finishes with a score of 0, that means the agent picked up the wrong item so make score negative
        if done and score == 0:
            score = -1

        return score, moves

    # Run the zork benchmork
    def testZork(self, agent):
        zorkPath = "./benchmark/zork1.z5"

        env = textworld.start(zorkPath)
        game_state = env.reset()
        reward, done, moves = 0, False, 0

        while not done and moves < self.max_moves:
            moves += 1
            command = agent.act(game_state, reward, done)
            print(game_state.feedback.strip())
            print(f"> {command}")
            print()
            game_state, reward, done = env.step(command)

        results = {
            "score" : game_state['score'],
            "moves" : moves
        }

        self.saveResults(results)
        pprint(results)

    def saveResults(self, results):
        if self.save_to is None:
            return None

        with open(self.save_to, "w") as file:
            file.write(json.dumps(results))


        return f"Saved results to {self.save_to}"

def main(args):
    benchmark = Benchmark()
    benchmark.save_to = args.save


    if args.generate:
        benchmark.generateGames()
        benchmark.registerGames()
        return 1

    if args.agent == "baseline":
        agent = BaselineAgent()
    elif args.agent == "rnn":
        agent = RNNAgent()
    elif args.agent == "transformer":
        agent = TransformerAgent()
    else:
        print("Agent not recognized")
        return 0

    benchmark.registerGames()

    if args.type == "zork":
        benchmark.testZork(agent)
    elif args.type == "treasure":
        benchmark.runBenchmark(agent)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Benchmarks")
    parser.add_argument("--generate", action="store_true",
                        help="Create the benchmark games")
    parser.add_argument(
        "--agent", choices=["baseline", "rnn", "transformer"], help="Which game agent to use")
    parser.add_argument(
        "--type", choices=["treasure", "zork"], help="Which kind of benchmark to run")
    parser.add_argument(
        "--save", type=str, help="JSON file to save the benchmark results in")

    args = parser.parse_args()
    main(args)
