# Solve system path
import sys
sys.path.append('./')

# Import libraries
import torch
from argparse import ArgumentParser

# Import local libraries
import knowledgeGraph.graph as graph
import training.templating as templating

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


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--slot_method", choices=["heuristic", "learned"], help="Choose a mode for the slot filler")
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

    main(p.parse_args())
