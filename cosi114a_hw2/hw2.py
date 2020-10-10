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
        self.distribution = distribution
        self.sorted_list = list()

    def probability_tuples(self):
        if not self.sorted_list:
            dec_dic = list()   # make a list of self.distribution copy
            for pair in self.distribution.items():
                dec_dic.append(pair)
            self.sorted_list = sorted(dec_dic, key=lambda t: t[1], reverse=True)  # in decreasing probability
        return self.sorted_list

    def sample(self):
        sorted_list = self.probability_tuples()
        # MSS sampling algorithm
        sum = 0.0
        rand = random.random()
        for i in sorted_list:
            prob_dist = i[1]
            sum += prob_dist
            if sum >= rand:
                return i[0]

# Problem 2: Sampling from bigram & trigram models
class BigramSampler(object):
    def __init__(
        self, distribution, start_token=START_TOKEN, end_token=END_TOKEN,
    ):
        self.distribution = distribution  # a nested dictionary
        self.start_token = start_token
        self.end_token = end_token

    def sample(self):
        str_output = ""  # separate token by spaces
        sampler_start = Sampler(self.distribution[self.start_token])
        token = sampler_start.sample()
        str_output += token + " "
        while not token == self.end_token:
            temp_sampler = Sampler(self.distribution[token])
            new_token = temp_sampler.sample()
            if not new_token == self.end_token:
                str_output += new_token + " "
            token = new_token
        return str_output.rstrip(" ")

class TrigramSampler(object):
    def __init__(
        self, distribution, start_token=START_TOKEN, end_token=END_TOKEN,
    ):
        self.distribution = distribution  # a nested dictionary
        self.start_token = START_TOKEN
        self.end_token = END_TOKEN

    def sample(self):
        str_output = ""  # separate token by spaces

        # token returned for <start> <start>
        sampler_start = Sampler(self.distribution[self.start_token, self.start_token])
        token_first = sampler_start.sample()
        str_output += token_first + " "
        # token returned for <start> token_first
        sampler_1 = Sampler(self.distribution[self.start_token, token_first])
        token_second = sampler_1.sample()
        if not token_second == self.end_token:
            str_output += token_second + " "

        while not token_second == self.end_token:
            temp_sampler = Sampler(self.distribution[token_first, token_second])
            new_token = temp_sampler.sample()
            if not new_token == self.end_token:
                str_output += new_token + " "
            token_first = token_second
            token_second = new_token
        return str_output.rstrip(" ")

# Problem 3: Sequence probability under bigram & trigram model
def sequence_probability_bigram(sentence, bigram_dist, start_token=START_TOKEN, end_token=END_TOKEN,):
    prob = 0.0
    # if a word in the sentence is not in the dictionary key, return prob = 0.0
    for i in range(len(sentence)):
        if sentence[i] not in bigram_dist:
            return prob

    for i in range(len(sentence)):
        if i == 0:
            # <start>
            prob += math.log(bigram_dist[start_token][sentence[i]])
        else:
            # other cases
            prob += math.log(bigram_dist[sentence[i-1]][sentence[i]])
    # <end>
    prob += math.log(bigram_dist[sentence[len(sentence)-1]][end_token])
    return math.exp(prob)

def sequence_probability_trigram(
    sentence, trigram_dist, start_token=START_TOKEN, end_token=END_TOKEN,
):
    prob = 0.0
    # <start> <start> first_word
    raw_prob = trigram_dist[start_token, start_token][sentence[0]]
    # if a word in the sentence is not in the dictionary key, return prob = 0.0
    if raw_prob == 0.0:
        return prob
    prob += math.log(raw_prob)
    for i in range(len(sentence)):
        if i == 0:
            # <start> first_word second_word
            raw_prob_start = trigram_dist[start_token, sentence[i]][sentence[i + 1]]
            if raw_prob_start == 0.0:
                return 0.0
            prob += math.log(raw_prob_start)
        elif i == len(sentence)-1:
            # second_last last <end>
            raw_prob_last = trigram_dist[sentence[i-1], sentence[i]][end_token]
            if raw_prob_last == 0.0:
                return 0.0
            prob += math.log(raw_prob_last)
        else:
            # first_word second_word third_word
            raw_prob_other = trigram_dist[sentence[i-1], sentence[i]][sentence[i+1]]
            if raw_prob_other == 0.0:
                return 0.0
            prob += math.log(raw_prob_other)
    # last <end> <end>
    raw_prob_end = trigram_dist[(sentence[len(sentence)-1], end_token)][end_token]
    if raw_prob_end == 0.0:
        return 0.0
    prob += math.log(raw_prob_end)
    return math.exp(prob)
