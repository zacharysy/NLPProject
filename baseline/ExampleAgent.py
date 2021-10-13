# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import numpy as np

import textworld
import textworld.challenges
from textworld.generator.game import GameOptions
import textworld.generator


class CustomAgent(textworld.Agent):
    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.actions = ["north", "south", "east", "west", "up", "down",
                        "look", "inventory", "take all", "YES", "wait",
                        "take", "drop", "eat", "attack"]

    def reset(self, env):
        env.display_command_during_render = True

    def act(self, game_state, reward, done):
        action = self.rng.choice(self.actions)
        if action in ["take", "drop", "eat", "attack"]:
            words = game_state.feedback.split()  # Observed words.
            words = [w for w in words if len(w) > 3]  # Ignore most stop words.
            if len(words) > 0:
                action += " " + self.rng.choice(words)

        return action

# gamestate: done, feedback, last_command, raw
if __name__ == "__main__":
    agent = CustomAgent()
#     env = textworld.start("zdungeon.z5")


    path = textworld.generator.compile_game(game)
    env = textworld.start(path)
    gameState = env.reset()
    reward, done = 0, False

    print(gameState.feedback)
    move = 0
    while not done and move < 1000:
        command = agent.act(gameState, reward, done)
        gameState, reward, done = env.step(command)
        print("> ", gameState.last_command, "\t score: ", gameState.score)
        print()
        print(gameState.feedback)
        move += 1

    print(move)

    env.reset()



