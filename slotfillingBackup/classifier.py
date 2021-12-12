import pymagnitude
import pathlib
import csv
import re
from pprint import pprint

EMBEDDING_PATH = pathlib.Path('training/glove_weights.magnitude')


'''
given a word, creates actions by filing in the templates found in templates.txt

has two functions:

`createActions(word) -> {"single": set, "double": set}`

given a word, returns a set of actions with that word filled in. "single" refers to actions that only require one object. "double" refers to actions which require two objects

"double" actions require passing in the action string to the next function with another word:

`fillAction(action, word)` -> String?

checks if the word can be placed in one of the action's slots. If it can, it will return the action with the word filled in. If not, this function returns None.
'''
class ActionGenerator:
	def __init__(self):
		self.attributes = ["portable","edible","moveable","switchable","flammable","openable","lockable","container","person","enemy"]
		self.vocab, self.scores = self.getVocab()
		self.actions = self.getActions()
		self.vectors = pymagnitude.Magnitude(EMBEDDING_PATH)
		self.scoreThreshold = 1


	def getVocab(self, path="slotfillingBackup/target_attribute_scores.csv"):
		words = []
		scores = {}

		with open(path, "r", encoding="utf-8-sig") as file:
			reader = csv.DictReader(file)

			for row in reader:
				noun = row['noun']
				words.append(noun)
				scores[noun] = {}

				for attribute in self.attributes:
					scores[noun][attribute] = int(row[attribute])

		return words, scores

	def getActions(self):
		actions = {}

		for attribute in self.attributes:
			actions[attribute] = set()

		for line in open("slotfillingBackup/templates.txt", "r"):
			foundAttributes = map(lambda x: x.replace("<","").replace(">", ""), re.findall("<\w+>", line))

			for attribute in foundAttributes:
				actions[attribute].add(line.strip())

		return actions

	def getScore(self, word):
		similar = self.vectors.most_similar_to_given(word, self.vocab)
		return similar, self.scores[similar]

	def createActions(self, word):
		similar , scores = self.getScore(word)
		possibleActions = { "single": set(), "double": set() }

		for attribute, score in scores.items():
			if score > self.scoreThreshold:
				for action in self.actions[attribute]:
					if len(re.findall("<\w+>", action)) == 1:
						slots = "single"
					else:
						slots = "double"

					action = action.replace(f"<{attribute}>", word)
					possibleActions[slots].add(action)


		return possibleActions

	def fillAction(self, action, word):
		_ , scores = self.getScore(word)

		slots = re.findall("<\w+>", action)

		bestSlot = None
		bestScore = 0

		for slot in slots:
			attribute = slot.replace("<", "").replace(">", "")
			score = scores[attribute]

			if score > bestScore:
				bestSlot = slot
				bestScore = score

		if bestSlot is not None:
			return action.replace(bestSlot, word)

		return None


if __name__ == "__main__":
	generator = ActionGenerator()

	word1 = "cow"
	word2 = "pizza"

	print(f"word1: {word1}, word2: {word2}\n")

	actions = generator.createActions(word1)

	for action in actions["single"]:
		print(action)

	if word2 is not None:
		for action in actions["double"]:
			action = generator.fillAction(action, word2)
			if action is not None:
				print(action)
