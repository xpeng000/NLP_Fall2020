# HW 2 Stubs
# Version 1.2
# 10/1/2020

from typing import DefaultDict
from collections import defaultdict
import pickle
import random  # Use random.random() and nothing else from random
import math

START_TOKEN = "<start>"
END_TOKEN = "<end>"


def pickle_dict(path: str, d: DefaultDict) -> None:
    """Converts defaultdict to a dictionary and serializes it to disk.

    NOTE: DO NOT ALTER THIS FUNCTION!
    """
    output = dict(d)

    for k in output:
        output[k] = dict(output[k])
    with open(path, "wb") as f:
        pickle.dump(output, f)


def load_dict(path: str) -> DefaultDict:
    """Loads a serialized dictionary from disk and converts it to a defaultdict.

    NOTE: DO NOT ALTER THIS FUNCTION!
    """
    with open(path, "rb") as f:
        regular_dict = pickle.load(f)

    output: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for key in regular_dict:
        output[key] = defaultdict(float, regular_dict[key])

    return output


# Problem 1: Sampling from a distribution


class Sampler(object):
    def __init__(self, distribution):
        # TODO: implement me
        pass

    def probability_tuples(self):
        # TODO: implement me
        pass

    def sample(self):
        # TODO: implement me
        pass


# Problem 2: Sampling from bigram & trigram models


class BigramSampler(object):
    def __init__(
        self, distribution, start_token=START_TOKEN, end_token=END_TOKEN,
    ):
        # TODO: implement me
        pass

    def sample(self):
        # TODO: implement me
        pass


class TrigramSampler(object):
    def __init__(
        self, distribution, start_token=START_TOKEN, end_token=END_TOKEN,
    ):
        # TODO: implement me
        pass

    def sample(self):
        # TODO: implement me
        pass


# Problem 3: Sequence probability under bigram & trigram model


def sequence_probability_bigram(
    sentence, bigram_dist, start_token=START_TOKEN, end_token=END_TOKEN,
):
    # TODO: implement me
    pass


def sequence_probability_trigram(
    sentence, trigram_dist, start_token=START_TOKEN, end_token=END_TOKEN,
):
    # TODO: implement me
    pass
