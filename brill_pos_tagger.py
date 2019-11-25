import json
import sys

from collections import Counter, defaultdict
from itertools import chain, groupby, islice, zip_longest
from operator import itemgetter
from typing import Set, Dict, List, Tuple, Iterable

from termcolor import colored


def load_data(fh) -> Iterable[Tuple[Tuple[str], Tuple[str]]]:
    """Return a list of examples. Each examples is a pair <words, labels>
    in which words and labels are two lists of strings
    """
    
    if fh.endswith(".json"):
        if isinstance(fh, str):
            fh = open(fh)

        return json.load(fh)
    elif fh.endswith(".conllu"):
        from universal_dependencies.conllu_io import pos_from_conllu
        return list(pos_from_conllu(fh, without_padding=True))
    elif fh.endswith(".pkl"):
        if isinstance(fh, str):
            fh = open(fh, "rb")

        data = pickle.load(fh)
        return [(words, pos) for _, words, pos in data]
    else:
        raise Exception


def test_pos_tagger(tagger, dataset, partial_labels=None, listener=None):
    """
    Returns
    -------
    - error rate, a float in [0, 1]
      the percentage of word that is not correctly labeled 
    - labeled_data, a List of List of Tuple[str, str, str]
      a list of labeled sentences. Each sentence is a list of triplet <word, gold_tag, predicted_tag>
    """
    
    bar = iter if listener is None else listener.iter

    n_errors = 0
    n_words = 0

    labeled_data = []
    if not partial_labels:
        partial_labels = [[] * len(dataset)]

    for (words, gold_tags), partial_tags in bar(zip_longest(dataset, partial_labels)):
        tags = tagger.tag(words, partial_tags)

        n_words += len(tags)
        n_errors += sum(1 for pred, gold in zip(tags,
                                               gold_tags) if pred != gold)

        labeled_data.append(list(zip(words, gold_tags, tags)))

    return n_errors / n_words, labeled_data


def validate_rules(rules):
    
    for rule in rules:

        if rule.keys() != {"argument", "kind", "tag_after", "tag_before", "name"}:
            print(rule)
            print(colored("can not identify one of the rule part", "red"))
            sys.exit(1)

        if rule["kind"] not in rule_methods:
            print(rule)
            print(colored("unknown rule kind", "red"))
            sys.exit(1)


class BrillPosTagger:

    def __init__(self, rules):
        validate_rules(rules)
        self.rules = rules
        self.rule_stats = defaultdict(int)
        
    @staticmethod
    def apply_rule(rule, idx, words, tags):
        rule_method = rule_methods[rule["kind"]]
        return rule_method(rule, idx, words, tags)

    @staticmethod
    def apply_tag_before(rule, idx, words, tags):
        if idx != 0 and tags[idx - 1] == rule["argument"] and tags[idx] == rule["tag_before"]:
            tags[pos] = rule["tag_after"]
            return True, tags
        else:
            return False, tags

    @staticmethod
    def apply_tag_after(rule, idx, words, tags):
        if idx < len(tags) - 1 and tags[idx + 1] == rule["argument"] and tags[idx] == rule["tag_before"]:
            tags[idx] = rule["tag_after"]
            return True, tags
        else:
            return False, tags

    @staticmethod
    def apply_word_before(rule, idx, words, tags):
        if pos != 0 and words[idx - 1] == rule["argument"] and tags[idx] == rule["tag_before"]:
            tags[idx] = rule["tag_after"]
            return True, tags
        else:
            return False, tags

    @staticmethod
    def apply_word_after(rule, idx, words, tags):
        if idx < len(tags) - 1 and words[idx + 1] == rule["argument"] and tags[idx] == rule["tag_before"]:
            tags[idx] = rule["tag_after"]
            return True, tags
        else:
            return False, tags

    def apply_starts_with_capital(rule, idx, words, tags):
        if words[idx][0].isupper() and tags[idx] == rule["tag_before"]:
            tags[idx] = rule["tag_after"]
            return True, tags
        else:
            return False, tags

    def tag(self, words: Iterable[str], partial_tags=None):

        predicted_tags = [self.most_frequent_pos.get(w, "NOUN") for w in words]

        for i in range(len(predicted_tags)):
            for rule in self.rules:
                has_been_applied, predicted_tags = BrillPosTagger.apply_rule(rule, i, words, predicted_tags)

                if has_been_applied:
                    self.rule_stats[rule["name"]] += 1
                    break

        return predicted_tags
        
    def train(self,
              examples: Iterable[Tuple[List[str], List[str]]]):

        examples = (zip(*e) for e in examples)
        data = groupby(sorted(chain.from_iterable(examples)), key=itemgetter(0))
        data = ((word, Counter(list(zip(*ex))[1])) for word, ex in data)
        self.most_frequent_pos  = {k: v.most_common(1)[0][0] for k, v in data}


rule_methods = {"tag_before": BrillPosTagger.apply_tag_before,
                "word_before": BrillPosTagger.apply_word_before,
                "word_after": BrillPosTagger.apply_word_after,
                "starts_with_upper": BrillPosTagger.apply_starts_with_capital,
                "tag_after": BrillPosTagger.apply_tag_after}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", required=True)
    parser.add_argument("--dump", action="store_true", default=False)
    parser.add_argument("--test", required=True)
    parser.add_argument("--color", action="store_true", default=False)
    
    args = parser.parse_args()

    tagger = BrillPosTagger(json.load(open(args.rules)))
    tagger.train(load_data("raw_data/fr_gsd-ud-train.conllu"))
    
    scores, labelled_data = test_pos_tagger(tagger, load_data(args.test))

    if args.dump:

        for ex in labelled_data:
            if args.color:
                print(" ".join(f"{}@{}@".format(word, gold if gold == predicted else '/'.join([gold, colored(predicted, 'red')])) for word, gold, predicted in ex))
            else:
                print(" ".join(f"{}@{}@".format(word, gold if gold == predicted else '/'.join([gold, predicted])) for word, gold, predicted in ex))
            print("-" * 10)

    print(scores)
    print(tagger.rule_stats)
