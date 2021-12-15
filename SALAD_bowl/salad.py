# Solve system path
import sys
sys.path.append('./')

# Import libraries
import torch
import textworld
import tqdm
from argparse import ArgumentParser

# Import local libraries
import training.templating as templating
import knowledgeGraph.graph as graph
from dqn.DQAgent import PA_DQAgent, bad_feedback
from dqn.util import ReplayMemory, ReplayMemoryStore, get_device, preprocess_line


def train(agent, slot_filler, kg, episodes, max_moves, game_path, output_weight_path):
    dims = 50
    batch_size = 64
    rho = 0.25
    max_size = 50000
    optim = torch.optim.Adam(agent.dqn.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    replay_store = ReplayMemoryStore(
        batch_size=batch_size, rho=rho, max_size=max_size)

    for episode in range(episodes):
        ep_reward = 0
        total_loss = 0
        env = textworld.start(game_path)
        game_state = env.reset()
        prev_reward = 0
        reward, done, moves = 0, False, 0
        desc = preprocess_line(game_state['raw'], start_symbol=CLS)
        unique_states = {' '.join(desc)}

        for t in tqdm.tqdm(range(max_moves)):
            action = agent.act(desc)

            # print('command: ', action)
            game_state, reward, done = env.step(' '.join(action))

            r = reward - prev_reward
            priority = 1 if r > 0 else 0

            feedback = preprocess_line(
                game_state['feedback'], start_symbol=CLS)
            feedback_str = ' '.join(feedback)

            if done:
                s = ['done']
                unique_states.add(feedback_str)
                print(feedback_str)
                print(r)

            is_bad_feedback = bad_feedback(feedback_str)
            if is_bad_feedback:
                r -= 0.1
                s = desc

            if feedback_str in unique_states:
                r -= 0.1

            if not done and not is_bad_feedback or feedback_str not in unique_states:
                r += 0.01 if reward < 1 else 10
                desc = feedback
                s = feedback
                unique_states.add(feedback_str)

            ep_reward += r

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

            if done:
                break

        print(
            f"EPISODE: {episode}\nDONE: {done}\nREWARD: {reward}\nN UNIQUE STATES: {len(unique_states)}\nEPSILON: {agent.epsilon}\nSYNTHETIC REWARD: {ep_reward}")

        if episode % 5 == 0:
            agent.update_target_network()

    agent.dqn.save(output_weight_path)


def main(args):

    # Load the slot filler
    if args.slot_method == 'heuristic':
        pass

    if args.slot_method == 'learned':
        slot_filler, _ = templating.load_model(args.slot_fill_csv_path,
                                               args.slot_fill_tsv_path,
                                               args.embedding_path,
                                               args.slot_fill_num_verb_clusters,
                                               args.slot_fill_num_prep_clusters,
                                               weight_path=args.slot_fill_weight_path)

    kg = graph.KnowledgeGraph()
    
    if args.agent_weight_path:
        salad_agent = PA_DQAgent(dqn_weights=args.agent_weight_path, 
            embedding_path=args.embedding_path, transitions=args.max_moves*args.episodes) 
    else:
        PA_DQAgent(embedding_path=args.embedding_path, transitions=args.max_moves*args.episodes)

    if args.train:
        train(salad_agent, slot_filler, kg, args.episodes, args.max_moves, args.game_path, args.output_weight_path)

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--slot_method", choices=["heuristic",
                   "learned"], help="Choose a mode for the slot filler")
    p.add_argument("--slot_fill_weight_path",
                   help="Path to weights for slot filler")
    p.add_argument("--slot_fill_csv_path",
                   help="Path to CSV data for slot filler",
                   default='./training/csv_data.csv')
    p.add_argument("--slot_fill_tsv_path",
                   help="Path to TSV data for slot filler",
                   default='./training/tsv_data.csv')
    p.add_argument("--slot_fill_num_verb_clusters",
                   help="Number of verb clusters slot filler",
                   type=int,
                   default=20)
    p.add_argument("--slot_fill_num_prep_clusters",
                   help="Number of preposition clusters slot filler",
                   type=int,
                   default=20)
    p.add_argument("--embedding_path",
                   help="Path to PyMagnitude embeddings",
                   default='./training/glove_weights.magnitude')
    p.add_argument("--agent_weight_path", help="Path to weights for DQN agent")
    p.add_argument("--train", action="store_true", help="Train")
    p.add_argument("--game_path", default="", help="Path to game")
    p.add_argument("--output_weight_path", default="dqn.pt", help="Output path for generating weights")
    p.add_argument("--episodes", default=300, help="Number of episodes to train on")
    p.add_argument("--max_moves", default=1000, help="Max number of moves the agent is allowed")

    main(p.parse_args())
