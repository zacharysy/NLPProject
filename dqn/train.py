import textworld
import sys
import torch
import tqdm
import matplotlib.pyplot as plt

from util import CLS, ReplayMemory, ReplayMemoryStore,  get_device, preprocess_line, read_data
from argparse import ArgumentParser
from DQAgent import MA_DQAgent, PA_DQAgent, bad_feedback


sys.path.append("./")
zorkPath = "../benchmark/zork1.z5"
maxMoves = 1000


def main(args):
    word_vocab, action_vocab = read_data()
    dims = 50
    batch_size = 64
    rho = 0.25
    max_size = 50000
    if args.model == 'max':
        agent = MA_DQAgent(word_vocab=word_vocab,
                           action_vocab=action_vocab, dims=dims)
    else:
        agent = PA_DQAgent(
            dims=dims, action_vocab=action_vocab, embedding_path=args.embpath,
            transitions=maxMoves*args.episodes, gamma=0.25)

    optim = torch.optim.Adam(agent.dqn.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    replay_store = ReplayMemoryStore(
        batch_size=batch_size, rho=rho, max_size=max_size)

    reward_history = []
    synthetic_reward_history = []

    for episode in range(args.episodes):
        ep_reward = 0
        total_loss = 0
        env = textworld.start(zorkPath)
        game_state = env.reset()
        prev_reward = 0
        reward, done, moves = 0, False, 0
        desc = preprocess_line(game_state['raw'], start_symbol=CLS)
        prev_feedback = ' '.join(desc)
        unique_states = {' '.join(desc)}

        for t in tqdm.tqdm(range(maxMoves)):
            action = agent.act(desc)

            # print('command: ', action)
            game_state, reward, done = env.step(' '.join(action))

            r = reward - prev_reward
            priority = 1 if r > 0 else 0

            feedback = preprocess_line(
                game_state['feedback'], start_symbol=CLS)

            if done:
                s = ['done']
                unique_states.add(' '.join(feedback))
                r += 100

            feedback_str = ' '.join(feedback)

            is_bad_feedback = bad_feedback(feedback_str)
            if feedback_str == prev_feedback or is_bad_feedback:
                r -= 0.2
                # r -= 10
                s = desc

            if feedback_str in unique_states:
                r -= 0.1
                # r -= 10

            if not is_bad_feedback or feedback_str not in unique_states:
                r += 0.01 if reward < 1 else 10
                desc = feedback
                s = feedback
                unique_states.add(feedback_str)

            # elif not bad_feedback(feedback):
            #     r += 0.1 if reward < 1 else 1
            #     desc = feedback
            #     s = feedback
            #     unique_states.add(' '.join(feedback))
            # elif ' '.join(feedback) == prev_feedback:
            #     r -= 0.2
            #     s = desc
            # else:
            #     # print('bad...')
            #     r = -0.2
            #     s = desc

            ep_reward += r
            prev_feedback = ' '.join(feedback)

            # print(feedback, r)
            replay_store.add(ReplayMemory(
                desc, action, r, s, priority))

            prev_reward = reward

            if t % 10 == 0 and len(replay_store.store) >= batch_size:
                train_mems = replay_store.mini_sample()
                for t in train_mems:
                    r = torch.tensor(t.r_t, device=get_device())
                    if t.s_next[0] == 'done':
                        y = r
                    y, prediction = agent.loss_args(r, t)
                    loss = loss_fn(y, prediction)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    total_loss += loss.detach().item()

        # with open('./log.txt', 'a+') as f:
        #     f.write(f"EPISODE: {episode}\n")
        #     states = set()
        #     for mem in replay_store.store:
        #         states.add(' '.join(mem.s_t))

        #     for state in states:
        #         for action in agent.action_vocab:
        #             enc = agent.dqn.encode(state.split(' '), action.split(' '))
        #             f.write(
        #                 f"{state[:30]}\t{action}\t{agent.dqn(enc)}\n")
        reward_history.append(reward)
        synthetic_reward_history.append(ep_reward)
        print(
            f"EPISODE: {episode}\nDONE: {done}\nREWARD: {reward}\nN UNIQUE STATES: {len(unique_states)}\nEPSILON: {agent.epsilon}\nSYNTHETIC REWARD: {ep_reward}")

        if episode % 5 == 0:
            agent.update_target_network()

    fig, axs = plt.subplots(2)
    fig.suptitle('Real (top) vs synthetic (bottom) reward over episodes')
    axs[0].plot(list(range(args.episodes)), reward_history)
    axs[1].plot(list(range(args.episodes)), synthetic_reward_history)
    plt.show()

    agent.dqn.save('./pa_dqn.pt')


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--episodes", type=int,
                   help="number of episodes to train for")
    p.add_argument("--model", type=str, help="type of dqn to use (max or per)")
    p.add_argument("--embpath", type=str, help="embedding path")
    main(p.parse_args())
