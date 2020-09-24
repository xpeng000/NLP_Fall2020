import sys

from collections import defaultdict, Counter
from typing import Tuple, List, DefaultDict

# generate sentences from the given file name
def gen_sentences(path):
    with open(path, encoding='utf8') as file:
        for line in file:
            if line.split(' ') and line.split(' ')[0]:
                #if split doesn't return empty array
                yield line.rstrip('\n').split(' ')

# generate an unigram that yields a list of strings
def gen_unigrams(sentences):
    for line in sentences:
        for word in line:
            yield word

# generate a bigram that yields tuples of strings, e.g. [(<start>, a), (a, b), (b, <end>)]
def gen_bigrams(sentences):
    for line in sentences:
        start = True
        for word in line:
            if start:
                yield '<start>', word
                start = False
            else:
                yield prev, word
            prev = word
        yield word, '<end>'

# generate a trigram that yields tuples of strings, e.g. [(<start>, <start>, a), (<start>, a, b), (a, b, <end>), ...]
def gen_trigrams(sentences):
    for line in sentences:
        start_one = True
        start_two = True
        if len(line) == 1:
            yield '<start>', '<start>', line[0]
            yield '<start>', line[0], '<end>'
            yield line[0], '<end>', '<end>'
        else:
            for i in range(len(line)):
                if start_one and start_two:
                    yield '<start>', '<start>', line[i]
                    start_one = False
                elif start_two:
                    yield '<start>', line[i - 1], line[i]
                    start_two = False
                else:
                    yield line[i - 2], line[i - 1], line[i]
            yield line[len(line) - 2], line[len(line) - 1], '<end>'
            yield line[len(line) - 1], '<end>', '<end>'



def count_unigrams(sentences, lower=False):
    uni_count: Counter[str] = Counter()
    for line in sentences:
        for word in line:
            if lower:
                uni_count[word.lower()] += 1
            else:
                uni_count[word] += 1
    return uni_count

def count_bigrams(sentences, lower=False):
    bigram_counter: Counter[Tuple[str, str]] = Counter()
    bigrams = list(gen_bigrams(sentences))
    for tuples in bigrams:
        if lower:
            word_one, word_two = tuples
            new_tuple = word_one.lower(), word_two.lower()
            bigram_counter[new_tuple] += 1
        else:
            bigram_counter[tuples] += 1
    return bigram_counter

def count_trigrams(sentences, lower=False):
    trigrams = list(gen_trigrams(sentences))
    trigram_counter: Counter[Tuple[str, str, str]] = Counter()

    for tuples in trigrams:
        if lower:
            word_one, word_two, word_three = tuples
            new_tuple = word_one.lower(), word_two.lower(), word_three.lower()
            trigram_counter[new_tuple] += 1
        else:
            trigram_counter[tuples] += 1
    return trigram_counter

def bigram_freq_dist(sentences, lower=False):
    bigram_dist = defaultdict(lambda: defaultdict(int))
    bigrams = list(gen_bigrams(sentences))
    for tuples in bigrams:
        word_one, word_two = tuples
        if lower:
            word_one = word_one.lower()
            word_two = word_two.lower()
        bigram_dist[word_one][word_two] += 1
    return bigram_dist

def trigram_freq_dist(sentences, lower=False):
    trigram_dist = defaultdict(lambda: defaultdict(int))
    trigrams = list(gen_trigrams(sentences))
    for tuples_mapping in trigrams:
        word_one, word_two, word_three = tuples_mapping
        if lower:
            word_one = word_one.lower()
            word_two = word_two.lower()
            word_three = word_three.lower()
        trigram_dist[(word_one,word_two)][word_three] += 1
    return trigram_dist

def unigram_probabilities(sentences, lower=False):
    unigram_prob = defaultdict(float)
    uni_count: Counter[str] = count_unigrams(sentences, lower=lower)
    total_count = 0
    for word in uni_count:
        total_count += uni_count[word]
    for word in uni_count:
        unigram_prob[word] = uni_count[word]/total_count
    return unigram_prob

def bigram_probabilities(sentences, lower=False):
    bigram_prob = defaultdict(lambda: defaultdict(float))
    bigram_count = bigram_freq_dist(sentences, lower=lower)
    for word_one in bigram_count:
        dist = bigram_count[word_one]
        total = sum(dist.values()) # gives a list of values
        for item in dist:
            bigram_prob[word_one][item] = bigram_count[word_one][item]/total
    return bigram_prob

def trigram_probabilities(sentences, lower=False):
    trigram_prob = defaultdict(lambda: defaultdict(float))
    trigram_count = trigram_freq_dist(sentences, lower=lower)
    for word_one in trigram_count:
        dist = trigram_count[word_one]
        total = sum(dist.values())  # gives a list of values
        for item in dist:
            trigram_prob[word_one][item] = trigram_count[word_one][item] / total
    return trigram_prob


def unigram_probabilities_difference(uni_probs1, uni_probs2, intersection_only=False):
    unigram_prob_diff = defaultdict(float)
    for word in uni_probs1:
        if intersection_only:
            if uni_probs2[word] and uni_probs1[word]:
                unigram_prob_diff[word] = uni_probs1[word] - uni_probs2[word]
        else:
            unigram_prob_diff[word] = uni_probs1[word] - uni_probs2[word]
    return unigram_prob_diff


def bigram_probabilities_difference(bi_probs1, bi_probs2, first, intersection_only=False,):
    bigram_prob_diff = defaultdict(float)
    if intersection_only:
        for word_one in bi_probs1[first]:
            if not bi_probs2[first][word_one] == 0.0:
                bigram_prob_diff[word_one] = bi_probs1[first][word_one] - bi_probs2[first][word_one]
    else:
        for word_one in bi_probs1[first]:
            bigram_prob_diff[word_one] = bi_probs1[first][word_one] - bi_probs2[first][word_one]
        for word_two in bi_probs2:
            if bigram_prob_diff[word_two] == 0.0:
                bigram_prob_diff[word_two] = bi_probs1[first][word_two] - bi_probs2[first][word_two]
    return bigram_prob_diff







