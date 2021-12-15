# Solve system path
import sys
sys.path.append('./')

from knowledgeGraph.graph import KnowledgeGraph

def generate_actions(text: str, kg: KnowledgeGraph, slot_filler):
    # Construct state
    kg.construct_state(text)

    # Parse kg for candidate nouns
    candidate_nouns = kg.create_nouns()

    # Turn candidate nouns into candidate phrases
    candidate_phrases = ["take all", "n", "s", "w", "e", "ne", "nw", "se", "sw", "u", "d", "inventory", "look"]

    for noun_set in candidate_nouns:
        # Heuristic slot filler
		if slot_filler.slot_fill_type == "heuristic":
			candidate_phrases += slot_filler.createActionSet(noun_set)

        # Learned slot filler
		else:
        	if len(noun_set) == 1:
            	slot_filler_out = slot_filler.get_full_sentence(noun_set, 5, mode='top5')

        	else:
            	slot_filler_out = slot_filler.get_full_sentence(noun_set[0] + ['<SEP>'] + noun_set[1], 5, mode='top5')

			candidate_phrases.append(slot_filler_out)

    return candidate_phrases
