import nltk
from nltk import CFG
from nltk.tree import Tree

#defining the grammer
grammar = CFG.fromstring("""
    S -> NP VP
    VP -> V NP | V NP PP
    PP -> P NP
    V -> "saw" | "ate" | "walked"
    NP -> "Rahil"  | "Bob" | Det N | Det N PP
    Det -> "a" | "an" | "the" | "my"|"his"
    N -> "dog" | "cat" | "telescope" | "park"| "Moon"| "terrace"
    P -> "in" | "on" | "by" | "with"| "from"
""")

#define a proper sentence
sentence = "Rahil saw the Moon with the telescope from his terrace".split()

#bottom-up parser
print("Bottom-Up Parsing:")
bottom_up_parser = nltk.ChartParser(grammar)
bottom_up_trees = []
for tree in bottom_up_parser.parse(sentence):
    print(tree)
    tree.pretty_print()
    bottom_up_trees.append(tree)
if bottom_up_trees:#if parsing is possible, then print
    for tree in bottom_up_trees:
        tree.draw()


print("Top-Down Parsing:")
top_down_parser = nltk.RecursiveDescentParser(grammar)
top_down_trees = []
try:
    for tree in top_down_parser.parse(sentence):
        print(tree)
        tree.pretty_print()
        top_down_trees.append(tree)
except ValueError as e:
    print(f"Error in parsing: {e}")
if top_down_trees:#if parsing is possible, then print
    for tree in top_down_trees:
        tree.draw()
