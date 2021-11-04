import textworld
import textworld.challenges
from textworld.generator.game import GameOptions
import textworld.generator

sys.path.append("./")
from baseline.nounverb import NounVerb
from baseline.utils import extract_nouns

class BaseAgent(textworld.Agent):
    def act(self, game_state, reward, done):
        return self.callModel(game_state.feedback)

    def callModel(self, text):
        pass

if __name__ == "__main__":
    pass
