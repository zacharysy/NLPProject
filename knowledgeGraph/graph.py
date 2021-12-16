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

    def __init__(self, num_prev_states: int = 5, inventory_size: int = 100):
        # Log inputs
        self.num_prev_states = num_prev_states
        self.inventory_size = inventory_size

        self.states = []
        self.inventory = []
        self.location = None
        self.prev_scene = None

        # Save client
        self.client = StanfordOpenIE(properties=self.OPENIE_PROPERTIES)

    def flush(self):
        self.states = []
        self.inventory = []
        self.location = None
        self.prev_scene = None

    def push_state(self, state):
        """
        Push the current state to the previous states
        """

        self.states = [state] + self.states
        self.states = self.states[:self.num_prev_states]

    def update_inventory(self, scene: str) -> bool:
        new_items = []

        for line in scene.split('\n'):
            if ':' in line:
                line_split = line.split(':')[0]
                if line_split not in self.inventory:
                    new_items.append(line_split)

        self.inventory = list(set(new_items)) + self.inventory
        self.inventory = self.inventory[:self.inventory_size]

        return len(new_items) > 0

    def construct_state(self, scene: str):
        """
        This method constructs a graph from a scene description.

        The annotations take the form:

        subject, relation, object

        This can be interpreted as an edge:

        subject ----(relation)----> object
        """

        # Handle simple state
        if self.prev_scene is None:
            self.prev_scene = scene

        if self.update_inventory(scene):
            scene = self.prev_scene

        for item in self.inventory:
            scene += f' You have {item}.'

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
                    self.location = words[2]

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

    def create_nouns(self):
        """
        Get all nouns or noun pairs from the graph/inventory
        """
        noun_pairs = []

        # Handle inventory first
        for item in self.inventory:
            noun_pairs.append((item,))

        # Handle the graph
        root_nodes = [self.location, 'you'] if self.location is not None else ['you']

        idx_to_entity, entity_to_idx, adj_mat = self.states[0]

        for root_node in root_nodes:
            if root_node in entity_to_idx:
                mat_row = adj_mat[entity_to_idx[root_node]]
                child_idxs = [i for i, item in enumerate(mat_row) if item is not None]
                children = [idx_to_entity[i] for i in child_idxs]
                noun_pairs += [(i, ) for i in children]

        return noun_pairs

if __name__ == '__main__':
    kg = KnowledgeGraph()
    text = "You've entered a basement. You try to gain information on your surroundings by using a technique called \"looking.\" You need an unguarded exit? You should try going east. You don't like doors? Why not try going north, that entryway is unguarded."
    kg.construct_state(text)
