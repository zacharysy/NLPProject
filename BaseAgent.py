import sys
import textworld
sys.path.append("./")
sys.path.append("./translation/")

import torch
import translation.transformer as transformer

class BaseAgent(textworld.Agent):
    def act(self, game_state, reward, done):
        return self.callModel(game_state.feedback)

    def callModel(self, text):
        return "open mailbox"


class TransformerAgent(textworld.Agent):
    """
    More than meets the eye
    """
    def __init__(self, weight_path='./translation/transformer.ckpt'):
        # Load model
        # self.model = transformer.load_model(weight_path)
        self.model = torch.load(weight_path)

    def act(self, game_state, reward, done):
        return self.callModel(game_state.feedback)

    def callModel(self, text):
        # Parse the input text
        text = transformer.preprocess_line(text)

        # Run the prediction
        prediction = transformer.predict(self.model, text)

        return ''.join(prediction)

if __name__ == "__main__":
    pass
