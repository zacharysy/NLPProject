import knowledgeGraph.graph
import torch
import sys
from argparse import ArgumentParser

sys.path.append('./')


def main(args):
    slot_filler_model = torch.load(args.slot_fill_weight_path)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "--slot_method", choices=["heuristic", "learned"], help="Choose a mode for the slot filler")
    p.add_argument("--slot_fill_weight_path",
                   help="Path to weights for slot filler")
    p.add_argument("--agent_weight_path",
                   help="Path to weights for DQN agent")
    main(p.parse_args())
