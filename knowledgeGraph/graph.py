"""
This file contains an implementation of a knowledge graph. It uses
OpenIE to parse the sentence to get SRO dependencies, and then builds
those dependencies into a graph.
"""

# Import libraries
from openie import StanfordOpenIE


class KnowledgeGraph:
    """
    This class contains the knowledge graph implementation.
    """

    OPENIE_PROPERTIES = {
        'openie.affinity_probability_cap': 2 / 3,
    }

    def __init__(self, num_prev_states: int = 5):
        # Log inputs
        self.num_prev_states = num_prev_states

        self.states = []

        # Save client
        self.client = StanfordOpenIE(properties=self.OPENIE_PROPERTIES)


    def push_state(self, state):
        """
        Push the current state to the previous states
        """

        self.states = [state] + self.states
        self.states = self.states[:self.num_prev_states]

    def construct_state(self, scene: str):
        """
        This method constructs a graph from a scene description.

        The annotations take the form:

        subject, relation, object

        This can be interpreted as an edge:

        subject ----(relation)----> object
        """

        # Run through the client
        annotations = self.client.annotate(scene)

        # No useful informations found
        if len(annotations) == 0:
            return

        # Perform some data cleaning on the graph
        relations = []
        for annotation in annotations:
            words = annotation['subject'], annotation['relation'], annotation['object']
            words = [i.lower() for i in words]

            # Substitute a few common words
            if words[0] == 'we':
                words[0] = 'you'

                if words[1] == 'are in':
                    words[1] = "'ve entered"

            relations.append(words)

        # Create mappings to and from entities
        entities = {i[0] for i in relations} | {i[2] for i in relations}
        idx_to_entity = list(entities)
        entity_to_idx = {val: i for i, val in enumerate(idx_to_entity)}

        adj_mat = [[None for _ in entities] for _ in entities]

        # Populate the adjacency matrix
        for s, r, o in relations:
            adj_mat[entity_to_idx[s]][entity_to_idx[o]] = r

        state = idx_to_entity, entity_to_idx, adj_mat
        self.push_state(state)

if __name__ == '__main__':
    kg = KnowledgeGraph()
    text = "You've entered a basement. You try to gain information on your surroundings by using a technique called \"looking.\" You need an unguarded exit? You should try going east. You don't like doors? Why not try going north, that entryway is unguarded."
    kg.construct_state(text)
