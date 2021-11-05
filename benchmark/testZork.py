import BaseAgent
from translation.transformer import *
from translation.rnn import *
import textworld
from jericho import *
import jericho.util
import sys
from pprint import pprint

sys.path.append("./")


zorkPath = "./benchmark/zork1.z5"

maxMoves = 1000
# agent = BaseAgent.BaseAgent()
# agent = BaseAgent.TransformerAgent()
agent = BaseAgent.RNNAgent()

env = textworld.start(zorkPath)
game_state = env.reset()
reward, done, moves = 0, False, 0

while not done and moves < maxMoves:
    moves += 1
    command = agent.act(game_state, reward, done)
    print(command)
    game_state, reward, done = env.step(command)


print(f"Score: {game_state['score']},  Moves: {moves}")
