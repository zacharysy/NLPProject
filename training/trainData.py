import os
import os.path
from jericho import *
import jericho.util

games = "./training/jericho-game-suite"
saveTo = "./training/train.txt"

with open(saveTo, "w") as train:
    for file in os.listdir(games):
        path = os.path.join(games, file)
        if os.path.isfile(path):
            env = FrotzEnv(path)

            observation, info = env.reset()
            walkthrough = env.get_walkthrough()

            for act in walkthrough:
                train.write(jericho.util.clean(observation))
                train.write(f"\n{act}\n")
                observation, reward, done, info = env.step(act)


