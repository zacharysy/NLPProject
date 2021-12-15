# Solve system path
import sys
sys.path.append('./')

from knowledgeGraph.graph import KnowledgeGraph

def generate_actions(text: str, kg: KnowledgeGraph, slot_filler):
    # Construct state
    kg.construct_state(text)

    # Parse kg for candidate nouns
    candidate_nouns = [(['sword'], )]

    # Turn candidate nouns into candidate phrases
    candidate_phrases = []
    for noun_set in candidate_nouns:
        if len(noun_set) == 1:
            slot_filler_out = slot_filler.get_full_sentence(noun_set)

        else:
            slot_filler_out = slot_filler.get_full_sentence(noun_set[0] + ['<SEP>'] + noun_set[1])

        noun_result = []
        for i in slot_filler_out:
            if type(i) is list:
                noun_result += i
            else:
                noun_result.append(i)

        candidate_phrases.append(noun_result)

    return candidate_phrases
