from __future__ import annotations
from collections import defaultdict
from pprint import pprint
import json, os, tqdm, sys

import textworld
from textworld.generator.game import GameOptions
from textworld.generator import compile_game

sys.path.append("./")
from baseline.ExampleAgent import CustomAgent

class Benchmark:
    def __init__(self):
        self.games_root = "./benchmark/games"
        self.levels = 30
        self.trials = 100
        self.max_moves = 1000

    # Create a single treasure_hunter game with the given level and saves it to self.games_root
    def createGame(self, level: int, trial: int) -> str:
        options = GameOptions()
        options.path = f"{self.games_root}/{level}/{trial}/"

        game = textworld.challenges.treasure_hunter.make(settings={"level": level}, options=options)

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
            game_directory[level] = {}

            for j in range(trials):
                trial = j + 1

                game = self.createGame(level, trial)
                game_directory[level][trial] = game

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
                            game = list(filter(lambda x: x.path.endswith(".ulx"), games))[0]
                            game_directory[level.name][trial.name] = game.path

        with open(os.path.join(folder_path, "games.json"), "w") as file:
            file.write(json.dumps(game_directory))

    # Run the benchmark up to the given `levels` and `trials` number of trials with at most `max_moves` number of moves per game
    def runBenchmark(self,
                    agent: textworld.Agent,
                    directory_path: Optional(str) = None,
                    levels: Optional(int) = None,
                    trials: Optional(int) = None,
                    max_moves: Optional(int) = None
                ):
        directory_path = directory_path or self.games_root
        levels = levels or self.levels
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

        for level in range(1, levels + 1):
            print(f"Level {level}")
            results[level] = {
                "average_score": 0,
                "average_moves": 0
            }

            for trial in tqdm.tqdm(range(1, trials + 1)):
                score, moves = self.runGame(agent, games[str(level)][str(trial)], max_moves)
                results[level]["average_score"] += score
                results[level]["average_moves"] += moves

            results[level]["average_score"] /= trials
            results[level]["average_moves"] /= trials

        print()
        pprint(results)

    # run a single game
    def runGame(self, agent: textworld.Agent, game_path: str, max_moves: int) -> (int,int):
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

if __name__ == "__main__":
    agent = CustomAgent()
    benchmark = Benchmark()
    benchmark.registerGames()
    benchmark.runBenchmark(agent)
