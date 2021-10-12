# NLP Project

## Baseline Outline

1. Gather a corpus of fiction novels stored in a text file where
   each line of the text file is one sentence from a novel
2. Use [spaCy](https://spacy.io/) to extract the main noun and
   verb from each sentence in the training data to create a
   reduced list of noun/verb pairs
3. Implement a lookup table which can transform a verb to its 
   present-tense form and use it to convert all verbs from
   step 2 into the present tense
4. Create a probability distribution similar to a bigram language
   model which learns `P(verb | noun)` from the corpus created
   in the previous steps
5. Implement an agent using the 
   [TextWorld](https://github.com/microsoft/textworld)
   library which iterates through the input for a given scenario,
   samples from `verb' = P(verb | noun)`, and attempts to submit
   `verb' noun` as a command for a given room
6. Test the agent on the 
   [TextWorld](https://github.com/microsoft/textworld)
   benchmark.
   
Step Breakdown: 

* Patrick Faley: 2 and 4
* Patrick Soga: 1 and 3
* Zachary Sy: 5 and 6
