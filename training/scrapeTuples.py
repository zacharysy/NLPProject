'''
parses all the sentences in `textPath` for verbs and saves them in `saveTo` with the ff format:
    <verb>, <object verb being done on>, <preposition>, <object doing verb>
or
    <verb>, <object verb being done on>,,
'''

# from baseline.utils import convert_verb
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from pprint import pprint
import re
import tqdm
import os
from diaparser.parsers import Parser
import sys
sys.path.append("./")


def get_children(tokens, id):
    children = []
    for token in tokens:
        if token["head"] == id:
            children.append(token)

    return children


def complete_obj(tokens, start):
    # Get object
    nounName = []

    nounToken = tokens[start-1]  # what is being objected to
    nounName.append(nounToken["form"])

    for child in get_children(tokens, int(nounToken["id"])):
        if child["deprel"] == "nmod":
            nounName += complete_nmod(tokens, int(child["id"]))

    noun = " ".join(nounName)

    # Get verb
    verbName = []
    verbToken = tokens[nounToken["head"] - 1]
    verbName.append(verbToken["form"])

    for child in get_children(tokens, int(verbToken["id"])):
        if child["deprel"] == "compound:prt":
            verbName += [child["form"]]

    verb = " ".join(verbName)

    return noun, verb


def complete_nmod(tokens, start):
    nounName = []
    nounToken = tokens[start-1]  # what is being objected to
    nounName.append(nounToken["form"])

    for child in get_children(tokens, int(nounToken["id"])):
        if child["deprel"] == "case":
            nounName = [child["form"]] + nounName

    return nounName


def complete_obl(tokens, start):
    nounName = []
    nounToken = tokens[start-1]  # what is being objected to
    nounName.append(nounToken["form"])
    noun, prep = None, None

    for child in get_children(tokens, int(nounToken["id"])):
        if child["deprel"] == "nmod":
            nounName += complete_nmod(tokens, int(child["id"]))

    noun = " ".join(nounName)

    for child in get_children(tokens, int(nounToken["id"])):
        if child["deprel"] == "case":
            prep = child["form"]

    return noun, prep


def parse_line(line, saveTo):
    try:
        sentence = parser.predict(line, text="en").sentences[0]
    except:
        return

    tokens = sentence.to_tokens()

    data = {
        "n1": None,
        "n2": None,
        "verb": None,
        "prep": None
    }

    if "obj" in sentence.rels:
        n1, verb = complete_obj(tokens, sentence.rels.index("obj") + 1)

        data["n1"] = n1
        data["verb"] = verb

    if "obl" in sentence.rels:
        n2, pp = complete_obl(tokens, sentence.rels.index("obl") + 1)
        data["n2"] = n2
        data["prep"] = pp

    # parsed_lines.append(data)
    if data['verb'] is not None:
        with open(saveTo, 'a+') as file:
            line = None
            if data['n2'] is not None and data['prep'] is not None:
                line = f"{data['verb']}, {data['n1']}, {data['prep']}, {data['n2']}"
            else:
                line = f"{data['verb']}, {data['n1']},,"

            file.write(f"{line}\n")


if __name__ == "__main__":

    # textPath = "training/1342-0.txt"
    textPath = "/home/psoga/Documents/wikipedia/split_more"
    saveTo = "/home/psoga/Documents/school/nlp/NLPProject/training/depTuples1.csv"

    for i, f in enumerate(os.scandir(textPath)):
        print(f'parsing {i}: {f.name}')
        with open(f, "r") as file:
            block = file.read()
        # Split lines by ., ?, !, and any of the previous followed by a quote.
        lines = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', block)

        parser = Parser.load(lang="en")
        parsed_line_data = []

        with tqdm.tqdm(len(lines)) as progress:
            with ThreadPoolExecutor(max_workers=10000) as executor:
                results = list(tqdm.tqdm(executor.map(
                    partial(parse_line, saveTo=saveTo), lines), total=len(lines)))
        # for pld in parsed_line_data:
        #     if pld['verb'] is not None:
        #         with open(saveTo, 'a+') as file:
        #             line = None
        #             if pld['n2'] is not None and pld['prep'] is not None:
        #                 line = f"{pld['verb']}, {pld['n1']}, {pld['prep']}, {pld['n2']}"
        #             else:
        #                 line = f"{pld['verb']}, {pld['n1']},,"
        #             file.write(f"{line}\n")

        #         file.write(f"{line}\n")
        # for line in lines:
        #     try:
        #         sentence = parser.predict(line, text="en").sentences[0]
        #     except Exception as E:
        #         print(E)
        #         continue

        #     tokens = sentence.to_tokens()

        #     data = {
        #         "n1": None,
        #         "n2": None,
        #         "verb": None,
        #         "prep": None
        #     }

        #     if "obj" in sentence.rels:
        #         n1, verb = complete_obj(tokens, sentence.rels.index("obj") + 1)

        #         data["n1"] = n1
        #         data["verb"] = verb

        #     if "obl" in sentence.rels:
        #         n2, pp = complete_obl(tokens, sentence.rels.index("obl") + 1)
        #         data["n2"] = n2
        #         data["prep"] = pp
        #     if data['verb'] is not None:
        #         with open(saveTo, 'a+') as file:
        #             line = None
        #             if data['n2'] is not None and data['prep'] is not None:
        #                 line = f"{data['verb']}, {data['n1']}, {data['prep']}, {data['n2']}"
        #             else:
        #                 line = f"{data['verb']}, {data['n1']},,"

        #             file.write(f"{line}\n")
        #             print('saving...', line)
