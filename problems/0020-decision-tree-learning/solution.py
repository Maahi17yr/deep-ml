import math
from collections import Counter


def learn_decision_tree(examples, attributes, target_attr):
    """
    Build a classification decision tree (ID3) using entropy and information gain.

    Parameters
    ----------
    examples : list of dicts mapping attribute name -> value (including target_attr)
    attributes : list of attribute names still available for splitting
    target_attr : name of the class label key

    Returns
    -------
    A class label (leaf), or a nested dict {attribute: {value: subtree, ...}}.
    """

    def entropy(rows):
        counts = Counter(r[target_attr] for r in rows)
        total = len(rows)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())

    def majority(rows):
        counts = Counter(r[target_attr] for r in rows)
        top = max(counts.values())
        return min(label for label, c in counts.items() if c == top)  # alphabetical tie-break

    def info_gain(rows, attr):
        total = len(rows)
        remainder = 0.0
        for value in set(r[attr] for r in rows):
            subset = [r for r in rows if r[attr] == value]
            remainder += len(subset) / total * entropy(subset)
        return entropy(rows) - remainder

    labels = set(r[target_attr] for r in examples)
    if len(labels) == 1:                     # pure node
        return labels.pop()
    if not attributes:                       # nothing left to split on
        return majority(examples)

    # Best attribute: highest gain, earliest in `attributes` on ties
    best_attr, best_gain = None, -1.0
    for attr in attributes:
        gain = info_gain(examples, attr)
        if gain > best_gain + 1e-12:
            best_attr, best_gain = attr, gain

    remaining = [a for a in attributes if a != best_attr]
    tree = {best_attr: {}}
    for value in sorted(set(r[best_attr] for r in examples)):
        subset = [r for r in examples if r[best_attr] == value]
        tree[best_attr][value] = learn_decision_tree(subset, remaining, target_attr)
    return tree