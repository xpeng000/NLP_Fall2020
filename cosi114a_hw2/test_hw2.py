# test_hw2.py
# Version 1.2
# 10/1/2020

import os
import random
import unittest
from collections import defaultdict
from typing import Tuple, DefaultDict

from grader import Grader, points
from hw2 import (
    Sampler,
    BigramSampler,
    TrigramSampler,
    load_dict,
    sequence_probability_bigram,
    sequence_probability_trigram,
)

# Set seed for random number generator
DEFAULT_SEED = 12345  # do not change me!

START_TOKEN0 = "<start>"
END_TOKEN0 = "<end>"


def load_brown_bigram_distribution() -> DefaultDict[str, DefaultDict[str, float]]:
    path = os.path.join("test_data", "brown_humor_bigram_dist.pkl")

    return load_dict(path)


def load_brown_trigram_distribution() -> DefaultDict[
    Tuple[str, str], DefaultDict[str, float]
]:
    path = os.path.join("test_data", "brown_humor_trigram_dist.pkl")

    return load_dict(path)


def load_dummy_bigram_distribution() -> DefaultDict[str, DefaultDict[str, float]]:
    path = os.path.join("test_data", "hw1_tokenized_text_3_bigram_dist.pkl")

    return load_dict(path)


def load_dummy_trigram_distribution() -> DefaultDict[
    Tuple[str, str], DefaultDict[str, float]
]:
    path = os.path.join("test_data", "hw1_tokenized_text_3_trigram_dist.pkl")

    return load_dict(path)


class SeedControlledTestCase(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(DEFAULT_SEED)


class TestSampler(SeedControlledTestCase):
    @points(1)
    def test_simple_dist_return_type(self) -> None:
        simple_dist = defaultdict(float, {"a": 0.5, "b": 0.4, "c": 0.1})
        sampler = Sampler(simple_dist)
        sample = sampler.sample()
        self.assertEqual(str, type(sample))

    @points(3)
    def test_probability_tuples_values(self):
        simple_dist = defaultdict(float, {"b": 0.4, "a": 0.5, "c": 0.1})
        sampler = Sampler(simple_dist)
        prob_tuples = sampler.probability_tuples()
        self.assertEqual([("a", 0.5), ("b", 0.4), ("c", 0.1)], prob_tuples)

    @points(1)
    def test_probability_tuples_cached(self):
        simple_dist = defaultdict(float, {"a": 0.5, "b": 0.4, "c": 0.1})
        sampler = Sampler(simple_dist)
        prob_tuples1 = sampler.probability_tuples()
        prob_tuples2 = sampler.probability_tuples()
        self.assertTrue(
            prob_tuples1 is prob_tuples2,
            "Same probability tuples should be returned every time due to caching",
        )

    @points(5)
    def test_simple_dist_one(self) -> None:
        simple_dist = defaultdict(float, {"a": 0.5, "b": 0.4, "c": 0.1})
        sampler = Sampler(simple_dist)
        sample = sampler.sample()
        expected_output = "a"
        self.assertEqual(expected_output, sample)

    @points(5)
    def test_simple_dist_ten(self) -> None:
        simple_dist = defaultdict(float, {"a": 0.5, "b": 0.4, "c": 0.1})
        sampler = Sampler(simple_dist)
        expected_output = ["a", "a", "b", "a", "a", "a", "b", "a", "a", "a"]
        sample = [sampler.sample() for _ in range(10)]
        self.assertEqual(expected_output, sample)


class TestBigramSampler(SeedControlledTestCase):
    @points(1)
    def test_bigram_sampler_type(self) -> None:
        bigram_dist = load_brown_bigram_distribution()
        bigram_sampler = BigramSampler(bigram_dist)
        random_sentence = bigram_sampler.sample()
        self.assertEqual(str, type(random_sentence))

    @points(7)
    def test_sample_bigram_single_sentence(self) -> None:
        bigram_dist = load_brown_bigram_distribution()
        bigram_sampler = BigramSampler(bigram_dist)
        random_sentence = bigram_sampler.sample()
        expected_sentence = "Welch was all over the man , and her to Welch was , the pets were always did Welch was interrupted many on the return , television , and bitterly , alas , and again , no dust anywhere and sitter to illustrate the coloring room of blood ."
        self.assertEqual(expected_sentence, random_sentence)

    @points(7)
    def test_sample_bigram_ten_sentences(self) -> None:
        bigram_dist = load_brown_bigram_distribution()
        bigram_sampler = BigramSampler(bigram_dist)

        # Sample 10 sentences using different random seeds
        random_sentences = []
        for seed in range(10):
            random.seed(seed)
            random_sentences.append(bigram_sampler.sample())

        expected_sentences = [
            "When I mistrusted the broken reputation for his state of Grazie was very sweet -- offered Barco reached the Catatonia .",
            "The incident , in an ascending scale , and , Miranda's dumping the phone , by stating bluntly : `` nut '' ; ; ; ; ; ; ; ; ; ; ; ; ; ;",
            "If Blanche .",
            "They had said `` personalities '' .",
            "They were playing quietly in that Hank and all the majority ; ; ; ;",
            "With quibs and bitterly , melted marshmallows .",
            "Said the procedure of the doors , aside from being whispered to link our house in a time .",
            "But with dressing , having paused to revive him which , and gibes , and less harmless `` chi-chi '' -- that Mary and had been murdering women had no greater precision he had been uncovered .",
            "It had been no particular significance to get tired of Scandal was the peace and less than five minutes with shovels , they gazed , dear , made his watch , apprehended for liquor .",
            "In fact , became a meeting .",
        ]
        self.assertEqual(expected_sentences, random_sentences)


class TestTrigramSampler(SeedControlledTestCase):
    @points(1)
    def test_trigram_sampler_type(self) -> None:
        trigram_dist = load_brown_trigram_distribution()
        trigram_sampler = TrigramSampler(trigram_dist)
        random_sentence = trigram_sampler.sample()
        self.assertEqual(str, type(random_sentence))

    @points(7)
    def test_sample_trigram_single_sentence(self) -> None:
        trigram_dist = load_brown_trigram_distribution()
        trigram_sampler = TrigramSampler(trigram_dist)
        random_sentence = trigram_sampler.sample()
        expected_sentence = "Welch was in cahoots with Marshall and would use his power as D.A. to drag every possible sensation into the phone ."
        self.assertEqual(expected_sentence, random_sentence)

    @points(7)
    def test_sample_trigram_ten_sentences(self) -> None:
        trigram_dist = load_brown_trigram_distribution()
        trigram_sampler = TrigramSampler(trigram_dist)

        # Sample 10 sentences using different random seeds
        random_sentences = []
        for seed in range(10):
            random.seed(seed)
            random_sentences.append(trigram_sampler.sample())

        expected_sentences = [
            "When I arrived at Viola's I was saved from making the decision as the trial progressed for any hint which might give me a lead as to where to shop , which had opened the doors of Los Angeles-Pasadena Society to her .",
            "The incident , aside from reflecting on Welch's political career , had all but wrecked his home life .",
            "If Blanche had smiled and said with only minimum ruefulness , `` so it is good to know why .",
            "They had honeymooned in Rome ; ;",
            "They were `` personalities '' .",
            "With quibs and gibes , the situation seemed to be almost shapeless , she would have it , even a room for mud .",
            "Said the digger .",
            "But with Welch's relentless pursuit of the Culture Forum on `` The Civic Spirit of the forum , was to be in the procedure of Justice was a nice day for a total stranger , be credited ? ?",
            "It had a small table on which she noted a vase of red rosebuds ; ;",
            "In fact , it was a sort of Gwen Cafritz to Francesca's Perle Mesta .",
        ]
        self.assertEqual(expected_sentences, random_sentences)


class TestSequenceProbabilityBigram(SeedControlledTestCase):
    @points(1)
    def test_sequence_probability_bigram_type(self) -> None:
        bigram_dist = load_brown_bigram_distribution()
        sentence = ["The", "cat", "ate", "the", "sandwhich", "."]
        p_sentence = sequence_probability_bigram(sentence, bigram_dist)
        self.assertEqual(float, type(p_sentence))

    @points(7)
    def test_sequence_probability_bigram_hw1_sentences(self) -> None:
        bigram_dist = load_dummy_bigram_distribution()
        sentences = [
            ["The", "cat", "ate", "the", "sandwhich", "."],
            ["The", "dog", "drank", "coffee", "."],
            ["The", "cat", "drank", "tea", "."],
            ["The", "squirrel", "ate", "peanuts", "."],
            ["The", "squirrel", "ate", "the", "pizza", "."],
            ["The", "dog", "drank", "tea", "."],
            ["The", "dog", "chased", "the", "cat", "."],
        ]
        expected_probs = [
            0.021164021164021145,
            0.0952380952380952,
            0.06349206349206346,
            0.0952380952380952,
            0.06349206349206349,
            0.19047619047619047,
            0.01587301587301586,
        ]

        for sent, expected_prob in zip(sentences, expected_probs):
            p = sequence_probability_bigram(sent, bigram_dist)
            self.assertAlmostEqual(expected_prob, p)

    @points(7)
    def test_sequence_probability_bigram_unseen_sentences(self) -> None:
        bigram_dist = load_dummy_bigram_distribution()
        unseen_sentences = [
            "Maksaako EU-jäsenyys Suomelle jo liikaa".split(),
            "12 eri alojen edustajaa kertoo miten EU vaikuttaa esimerkiksi Suomen ruokavalikoimaan ja valkoselkätikkaan".split(),
            "Mit einem einzelnen Durchgreifen mit einer Studie oder mit einem schlauen Gedanken allein ist es jedenfalls nicht zu meistern".split(),
            "Messen Sie mit zweierlei Maß".split(),
            "русский сложный язык".split(),
            "τα ελληνικά είναι μια δύσκολη γλώσσα".split(),
        ]

        for sent in unseen_sentences:
            p = sequence_probability_bigram(sent, bigram_dist)
            self.assertAlmostEqual(0.0, p)


class TestSequenceProbabilityTrigram(SeedControlledTestCase):
    @points(1)
    def test_sequence_probability_trigram_type(self) -> None:
        trigram_dist = load_brown_trigram_distribution()
        sentence = ["The", "cat", "ate", "the", "sandwhich", "."]
        p_sentence = sequence_probability_trigram(sentence, trigram_dist)
        self.assertEqual(float, type(p_sentence))

    @points(7)
    def test_sequence_probability_trigram_hw1_sentences(self) -> None:
        trigram_dist = load_dummy_trigram_distribution()
        sentences = [
            ["The", "cat", "ate", "the", "sandwhich", "."],
            ["The", "dog", "drank", "coffee", "."],
            ["The", "cat", "drank", "tea", "."],
            ["The", "squirrel", "ate", "peanuts", "."],
            ["The", "squirrel", "ate", "the", "pizza", "."],
            ["The", "dog", "drank", "tea", "."],
            ["The", "dog", "chased", "the", "cat", "."],
        ]
        expected_probs = [
            0.07142857142857141,
            0.14285714285714285,
            0.14285714285714285,
            0.14285714285714285,
            0.07142857142857141,
            0.14285714285714285,
            0.14285714285714285,
        ]

        for sent, expected_prob in zip(sentences, expected_probs):
            p = sequence_probability_trigram(sent, trigram_dist)
            self.assertAlmostEqual(expected_prob, p)

    @points(7)
    def test_sequence_probability_trigram_unseen_sentences(self) -> None:
        trigram_dist = load_dummy_trigram_distribution()
        unseen_sentences = [
            "Maksaako EU-jäsenyys Suomelle jo liikaa".split(),
            "12 eri alojen edustajaa kertoo miten EU vaikuttaa esimerkiksi Suomen ruokavalikoimaan ja valkoselkätikkaan".split(),
            "Mit einem einzelnen Durchgreifen mit einer Studie oder mit einem schlauen Gedanken allein ist es jedenfalls nicht zu meistern".split(),
            "Messen Sie mit zweierlei Maß".split(),
            "русский сложный язык".split(),
            "τα ελληνικά είναι μια δύσκολη γλώσσα".split(),
        ]

        for sent in unseen_sentences:
            p = sequence_probability_trigram(sent, trigram_dist)
            self.assertAlmostEqual(0.0, p)


def main() -> None:
    tests = [
        TestSampler,
        TestBigramSampler,
        TestTrigramSampler,
        TestSequenceProbabilityBigram,
        TestSequenceProbabilityTrigram,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
