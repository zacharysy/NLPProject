import textworld
import sys
import torch

from random import choice
from typing import List
from dataclasses import dataclass
from random import randint, random
from util import CLS, ReplayMemory, Vocab, get_device, preprocess_line, read_data
from argparse import ArgumentParser
from DQAgent import MA_DQAgent, PA_DQAgent
from pprint import pprint


sys.path.append("./")
zorkPath = "../benchmark/zork1.z5"
maxMoves = 1000


def random_sample(iterable, n):
    len_iter = len(iterable)
    return [choice(iterable) for _ in range(n if n <= len_iter else len_iter)]


def mini_sample(rho, replay_memories: List[ReplayMemory]) -> List[ReplayMemory]:
    num_replay_mems = len(replay_memories)
    high_rhos, low_rhos = [], []
    for mem in replay_memories:
        (high_rhos if mem.p_t == 1 else low_rhos).append(mem)
    high_rho_ratio = len(list(high_rhos))/num_replay_mems

    if high_rho_ratio <= rho:
        return random_sample(replay_memories, 64)

    target = int(len(num_replay_mems) * high_rho_ratio)
    train_memories = random_sample(
        high_rhos, target) + random_sample(low_rhos, 64 - target)
    return train_memories


def bad_feedback(feedback):
    f = ' '.join(feedback)
    return "you don't" in f or "you can't" in f or "i don't" in f or \
        "i can't" in f or "?" in f or len(feedback) < 10


def main(args):
    word_vocab, action_vocab = read_data()
    dims = 200
    epsilon = 0.8
    rho = 0.25
    gamma = 0.5
    decay = 0.9999
    if args.model == 'max':
        agent = MA_DQAgent(word_vocab=word_vocab,
                           action_vocab=action_vocab, dims=dims, epsilon=epsilon)
    else:
        agent = PA_DQAgent(
            dims=dims, action_vocab=action_vocab, embedding_path=args.embpath, epsilon=epsilon)

    replay_mems: List[ReplayMemory] = []

    optim = torch.optim.Adam(agent.dqn.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for ep in range(args.episodes):
        print('episode: ', ep)
        total_loss = 0
        env = textworld.start(zorkPath)
        game_state = env.reset()
        prev_reward = 0
        reward, done, moves = 0, False, 0
        desc = preprocess_line(game_state['raw'], start_symbol=CLS)
        agent.set_epsilon()

        for t in range(maxMoves):
            action = agent.callModel(desc)

            print('command: ', action)
            game_state, reward, done = env.step(action)

            r = reward - prev_reward
            priority = 1 if r > 0 else 0

            feedback = preprocess_line(
                game_state['feedback'], start_symbol=CLS)

            if done:
                s = 'done'
            elif not bad_feedback(feedback):
                r = 1
                desc = feedback
                s = feedback
            else:
                print('bad...')
                r = -0.1
                s = desc

            print(feedback, r)
            replay_mems.append(ReplayMemory(
                desc, action, r, s, priority))

            prev_reward = reward
            epsilon *= decay

            if t % 4 == 0:
                train_mems = mini_sample(rho, replay_mems)
                for t in train_mems:
                    r = torch.tensor(t.r_t, device=get_device())
                    if t.s_next == 'done':
                        y = r
                    y, prediction = agent.loss_args(r, t)
                    loss = loss_fn(y, prediction)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    total_loss += loss.detach().item()
            print()


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--episodes", type=int,
                   help="number of episodes to train for")
    p.add_argument("--model", type=str, help="type of dqn to use (max or per)")
    p.add_argument("--embpath", type=str, help="embedding path")
    main(p.parse_args())
