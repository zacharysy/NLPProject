# Solve system path
import sys
sys.path.append('./')

from knowledgeGraph.graph import KnowledgeGraph

def generate_actions(text: str, kg: KnowledgeGraph, slot_filler):
    # Construct state
    kg.construct_state(' '.join(text))

    # Parse kg for candidate nouns
    candidate_nouns = kg.create_nouns()

    # Turn candidate nouns into candidate phrases
    candidate_phrases = ["take all", "n", "s", "w", "e", "ne", "nw", "se", "sw", "u", "d", "inventory", "look"]

    for noun_set in candidate_nouns:
        # Heuristic slot filler
        if slot_filler.slot_fill_type == "heuristic":
            candidate_phrases += slot_filler.createActionSet([i.split(' ') for i in noun_set])

        # Learned slot filler
        else:
            if len(noun_set) == 1:
                slot_filler_out = slot_filler.get_full_sentence(noun_set[0].split(' '), 5, mode='top5')

            else:
                slot_filler_out = slot_filler.get_full_sentence(noun_set[0].split(' ') + ['<SEP>'] + noun_set[1].split(' '), 5, mode='top5')
            slot_filler_out = [' '.join(i) for i in slot_filler_out]

            candidate_phrases += slot_filler_out

    return candidate_phrases
