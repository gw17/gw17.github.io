# ---------

grammar = (
    "S",
    [("T", "a"), ("Y", "b"), ("Z", "b")],
    [("S", "X", "Y"), ("T", "Z", "T"), ("X", "T", "Y"),
     ("Y", "Y", "T"), ("Z", "T", "Z")])

add_grammar = (
    "S",
    [("P", "+")] + [("A", str(i)) for i in range(10)] + [("B", str(i)) for i in range(10)],
    [("S", "A", "B"), ("B", "P", "A"), ("B", "P", "B")])
# ---------

from operator import itemgetter
from itertools import groupby, chain
from random import choice


def generate(grammar):

    rewriting_rules = groupby(sorted(grammar[1] + grammar[2]), #!\label{line:firstpart_start}!
                              key=itemgetter(0))
    rewriting_rules = {rhs: [g[1:] for g in group]
                       for rhs, group in rewriting_rules}      #!\label{line:firstpart_end}!

    proto_mot = [grammar[0]] # !\label{line:secondpart_start}!
    while any(w.isupper() for w in proto_mot):
        proto_mot = (choice(rewriting_rules[w]) if w.isupper() else w
                     for w in proto_mot) # !\label{line:gen}!
        proto_mot = list(chain.from_iterable(proto_mot)) # !\label{line:secondpart_end}!

    return proto_mot

# ---------

from collections import defaultdict

def recognize(word, grammar):

    axiom, term_prod, non_term_prod = grammar
    
    chart = defaultdict(set)

    for i, w in enumerate(word):
        chart[i, i].update(nt for nt, t in term_prod if t == w)

    for d in range(1, len(word)):
        for i in range(0, len(word) - d):
            j = i + d
            for k in range(i, j):
                for lhs, rhs1, rhs2 in non_term_prod:
                    if rhs1 in chart[i, k] and rhs2 in chart[k + 1, j]:
                        chart[i, j].add(lhs)

    return axiom in chart[0, len(word) - 1]

# ---------

from collections import namedtuple, defaultdict

Item = namedtuple("Item", "rhs lhs k")

def parse(word, grammar):

    axiom, term_prod, non_term_prod = grammar

    chart = defaultdict(dict)

    for i, w in enumerate(word):
        chart[i, i] = {non_term: Item(lhs=non_term, rhs=term, k=0)
                       for non_term, term in term_prod
                       if term == w}

    for d in range(1, len(word)):
        for i in range(0, len(word) - d):
            j = i + d
            for k in range(i, j):
                for lhs, rhs1, rhs2 in non_term_prod:
                    if rhs1 in chart[i, k] and rhs2 in chart[k + 1, j]:
                        chart[i, j][lhs] = Item(lhs=lhs,
                                                rhs=(rhs1, rhs2),
                                                k=k)

    return chart
# ---------

def build_tree(chart, axiom, word):
    
    def build_tree_rec(chart, non_terminal, i, j):
        item = [it for it in chart[i, j].values()
                if it.lhs == non_terminal][0]

        if type(item.rhs) == type(""):
            return (non_terminal, item.rhs)

        return (non_terminal,
                build_tree_rec(chart, item.rhs[0], i, item.k),
                build_tree_rec(chart, item.rhs[1], item.k + 1, j))

    return build_tree_rec(chart, axiom, 0, len(word) - 1)

# ---------

def count_derivations(word, grammar):

    axiom, term_prod, non_term_prod = grammar
    
    chart = defaultdict(lambda: defaultdict(int))

    for i, w in enumerate(word):
        for nt in (nt for nt, t in term_prod if t == w):
            chart[i, i][nt] += 1

    for d in range(1, len(word)):
        for i in range(0, len(word) - d + 1):
            j = i + d
            for k in range(i, j):
                for lhs, rhs1, rhs2 in non_term_prod:
                    if rhs1 in chart[i, k] and rhs2 in chart[k + 1, j]:
                        cnt = chart[i, k][rhs1] * chart[k + 1, j][rhs2]
                        chart[i, j][lhs] += cnt

    return chart[0, len(word) - 1][axiom]

# -----
from collections import defaultdict, namedtuple

WItem = namedtuple("WItem", "rhs lhs k weight")

def best_derivation(word, grammar):

    axiom, term_prod, non_term_prod = grammar

    chart = defaultdict(dict)

    for i, w in enumerate(word):
        chart[i, i] = {non_term: WItem(lhs=non_term, rhs=term, k=0, weight=weight)
                       for non_term, term, weight in term_prod
                       if term == w}
    print(chart)
    for d in range(1, len(word)):
        for i in range(0, len(word) - d):
            j = i + d
            for k in range(i, j):
                for lhs, rhs1, rhs2, weight in non_term_prod:
                    w = weight +\
                        chart[i, k][rhs1].weight if rhs1 in chart[i, k]\
                          else float("-infinity") +\
                        chart[k + 1, j][rhs2].weight if rhs2 in chart[k + 1, j]\
                          else float("-infinity")])

                    if lhs not in chart[i, j] or w > chart[i, j][lhs].weight:
                        chart[i, j][lhs] = WItem(lhs=lhs,
                                                 rhs=(rhs1, rhs2),
                                                 k=k,
                                                 weight=w)

    return chart

from math import log10

wcfg = ["S",
        (("P", "with", log10(1)),
         ("V", "saw", log10(1)),
         ("NP", "astronomers", log10(.1)),
         ("NP", "ears", log10(0.18)),
         ("NP", "saw", log10(0.04)),
         ("NP", "stars", log10(0.18)),
         ("NP", "telescopes", log10(0.1))),
        (("S", "NP", "VP", log10(1)),
         ("PP", "P", "NP", log10(1)),
         ("VP", "V", "NP", log10(0.7)),
         ("VP", "VP", "PP", log10(0.3)),
         ("NP", "NP", "PP", log10(0.4)))]


word = "astronomers saw stars with ears".split()
chart = best_derivation(word, wcfg)

print(f"best proba: {10 ** chart[0, len(word) - 1]['S'].weight:.7}")
print(build_tree(chart,
                 "S",
                 word))
