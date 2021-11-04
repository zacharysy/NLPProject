import sys
import textworld
sys.path.append("./")

class BaseAgent(textworld.Agent):
    def act(self, game_state, reward, done):
        return self.callModel(game_state.feedback)

    def callModel(self, text):
        return "open mailbox"

if __name__ == "__main__":
    pass
