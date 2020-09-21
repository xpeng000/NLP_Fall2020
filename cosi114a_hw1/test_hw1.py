#! /usr/bin/env python

"""
Public tests for HW 1

Version 1.1, released 9/14/2020
"""

import os
import unittest
from collections import defaultdict
from types import GeneratorType

from grader import Grader, points

from hw1 import (
    gen_unigrams,
    gen_sentences,
    gen_bigrams,
    gen_trigrams,
    count_unigrams,
    count_bigrams,
    count_trigrams,
    bigram_freq_dist,
    trigram_freq_dist,
    unigram_probabilities,
    bigram_probabilities,
    trigram_probabilities,
    unigram_probabilities_difference,
    bigram_probabilities_difference,
)

START_TOKEN0 = "<start>"
END_TOKEN0 = "<end>"

START_TOKEN1 = "<start>"
END_TOKEN1 = "<end>"


class TestGenSentences(unittest.TestCase):
    @points(1)
    def test_all(self) -> None:
        """Test all of gen_sentences."""
        # Test type
        gen = gen_sentences(os.path.join("test_data", "hw1_tokenized_text_1.txt"))
        self.assertEqual(GeneratorType, type(gen))

        # Test basic
        gen = gen_sentences(os.path.join("test_data", "hw1_tokenized_text_1.txt"))
        self.assertEqual(
            ["Tokenized", "text", "is", "easy", "to", "work", "with", "."], next(gen)
        )
        self.assertEqual(
            ["Writing", "a", "tokenizer", "is", "a", "pain", "."], next(gen)
        )
        with self.assertRaises(StopIteration):
            next(gen)

        # Test advanced
        gen = gen_sentences(os.path.join("test_data", "hw1_tokenized_text_2.txt"))
        self.assertEqual(["Hello", ",", "world", "!"], next(gen))
        # Between these sentences, there is a line in the file with a single space,
        # which should be skipped over.
        self.assertEqual(["This", "is", "a", "normal", "sentence", "."], next(gen))
        self.assertEqual(
            [
                '"',
                "I",
                "don't",
                "like",
                "it",
                "when",
                "there's",
                "too",
                "much",
                "punctuation",
                "!",
                '"',
                ",",
                "they",
                "exclaimed",
                ".",
            ],
            next(gen),
        )
        with self.assertRaises(StopIteration):
            next(gen)


class TestNGrams(unittest.TestCase):
    @points(1)
    def test_type_unigram(self) -> None:
        """Test that a generator is returned."""
        gen = gen_unigrams(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_1.txt"))
        )
        self.assertEqual(GeneratorType, type(gen))
        self.assertEqual(str, type(next(gen)))

    @points(1)
    def test_type_bigrams(self) -> None:
        gen = gen_bigrams(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_1.txt"))
        )
        self.assertEqual(GeneratorType, type(gen))
        self.assertEqual(tuple, type(next(gen)))

    @points(1)
    def test_type_trigrams(self) -> None:
        gen = gen_trigrams(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_1.txt"))
        )
        self.assertEqual(GeneratorType, type(gen))
        self.assertEqual(tuple, type(next(gen)))

    @points(5)
    def test_unigrams(self) -> None:
        """Basic unigram test"""
        self.assertEqual(
            ["Hello", ",", "world", "!"],
            list(gen_unigrams([["Hello", ",", "world", "!"]])),
        )

    @points(5)
    def test_bigrams(self) -> None:
        """Basic bigram test"""
        self.assertEqual(
            [
                (START_TOKEN0, "Hello"),
                ("Hello", ","),
                (",", "world"),
                ("world", "!"),
                ("!", END_TOKEN0),
            ],
            list(gen_bigrams([["Hello", ",", "world", "!"]])),
        )

    @points(5)
    def test_trigrams(self) -> None:
        """Basic trigram test"""
        self.assertEqual(
            [
                (START_TOKEN1, START_TOKEN0, "Hello"),
                (START_TOKEN0, "Hello", ","),
                ("Hello", ",", "world"),
                (",", "world", "!"),
                ("world", "!", END_TOKEN0),
                ("!", END_TOKEN0, END_TOKEN1),
            ],
            list(gen_trigrams([["Hello", ",", "world", "!"]])),
        )

    @points(1)
    def test_bigram_multisentence(self) -> None:
        self.assertEqual(
            [
                (START_TOKEN0, "Hello"),
                ("Hello", ","),
                (",", "world"),
                ("world", "!"),
                ("!", END_TOKEN0),
                (START_TOKEN0, "Good"),
                ("Good", "bye"),
                ("bye", END_TOKEN0),
            ],
            list(gen_bigrams([["Hello", ",", "world", "!"], ["Good", "bye"]])),
        )

    @points(1)
    def test_trigrams_multisentence(self) -> None:
        """Basic trigram test"""
        self.assertEqual(
            [
                (START_TOKEN1, START_TOKEN0, "Hello"),
                (START_TOKEN0, "Hello", ","),
                ("Hello", ",", "world"),
                (",", "world", "!"),
                ("world", "!", END_TOKEN0),
                ("!", END_TOKEN0, END_TOKEN1),
                (START_TOKEN1, START_TOKEN0, "Good"),
                (START_TOKEN0, "Good", "bye"),
                ("Good", "bye", END_TOKEN0),
                ("bye", END_TOKEN0, END_TOKEN1),
            ],
            list(gen_trigrams([["Hello", ",", "world", "!"], ["Good", "bye"]])),
        )


class TestCounts(unittest.TestCase):
    @points(1)
    def test_count_unigrams_type(self) -> None:
        """Test count unigrams type"""
        gen = gen_sentences(os.path.join("test_data", "hw1_tokenized_text_1.txt"))
        counts = count_unigrams(gen)
        for k in counts:
            self.assertEqual(str, type(k))

    @points(2)
    def test_count_unigrams(self) -> None:
        """Test count unigrams with case"""
        unigrams = count_unigrams(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt"))
        )

        self.assertEqual(7, unigrams["The"])
        self.assertEqual(3, unigrams["dog"])
        self.assertEqual(
            3, unigrams["cat"],
        )
        self.assertEqual(3, unigrams["the"])
        self.assertEqual(7, unigrams["."])
        self.assertEqual(1, unigrams["pizza"])

    @points(1)
    def test_count_unigrams_lower(self) -> None:
        """Test count unigrams with lowercase option=True"""
        unigrams = count_unigrams(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt")),
            lower=True,
        )
        self.assertEqual(10, unigrams["the"])
        self.assertEqual(3, unigrams["cat"])
        self.assertEqual(3, unigrams["dog"])
        self.assertEqual(7, unigrams["."])
        self.assertEqual(1, unigrams["pizza"])

    @points(1)
    def test_count_bigrams_type(self) -> None:
        """Test bigrams are tuples with 2 strings"""
        # assert case_sarcastically("hello, friend!") == "hElLo, FrIeNd!"
        gen = gen_sentences(os.path.join("test_data", "hw1_tokenized_text_1.txt"))
        counts = count_bigrams(gen)
        for k in counts:
            self.assertEqual(tuple, type(k))
            self.assertEqual(2, len(k))

    @points(1)
    def test_count_bigrams(self) -> None:
        """Test count bigrams with case"""
        bigrams = count_bigrams(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt"))
        )

        self.assertEqual(3, bigrams[("The", "dog")])
        self.assertEqual(2, bigrams[("squirrel", "ate")])
        self.assertEqual(2, bigrams[("The", "cat")])
        self.assertEqual(1, bigrams[("the", "cat")])
        self.assertEqual(1, bigrams[("drank", "coffee")])
        self.assertEqual(7, bigrams[".", END_TOKEN0])

    @points(1)
    def test_count_bigrams_lower(self) -> None:
        """Test count bigrams with lowercase"""
        bigrams = count_bigrams(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt")),
            lower=True,
        )

        self.assertEqual(3, bigrams[("the", "dog")])
        self.assertEqual(2, bigrams[("squirrel", "ate")])
        self.assertEqual(0, bigrams[("The", "cat")])
        self.assertEqual(3, bigrams[("the", "cat")])
        self.assertEqual(1, bigrams[("drank", "coffee")])
        self.assertEqual(7, bigrams[".", END_TOKEN0])

    @points(1)
    def test_count_trigrams_type(self) -> None:
        """Test trigrams are tuples with 3 strings"""
        # assert case_sarcastically("hello, friend!") == "hElLo, FrIeNd!"
        gen = gen_sentences(os.path.join("test_data", "hw1_tokenized_text_1.txt"))
        counts = count_trigrams(gen)
        for k in counts:
            self.assertEqual(tuple, type(k))
            self.assertEqual(3, len(k))

    @points(1)
    def test_count_trigrams(self) -> None:
        """Test count trigrams with casing."""
        trigrams = count_trigrams(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt"))
        )
        self.assertEqual(2, trigrams[("The", "dog", "drank")])
        self.assertEqual(1, trigrams[("squirrel", "ate", "peanuts")])
        self.assertEqual(7, trigrams[(".", END_TOKEN0, END_TOKEN1)])
        self.assertEqual(7, trigrams[(START_TOKEN1, START_TOKEN0, "The")])

    @points(1)
    def test_count_trigrams_lower(self) -> None:
        """Test count trigrams with lowercasing."""
        trigrams = count_trigrams(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt")),
            lower=True,
        )
        self.assertEqual(0, trigrams[(START_TOKEN1, START_TOKEN0, "The")])
        self.assertEqual(7, trigrams[(START_TOKEN1, START_TOKEN0, "the")])


class TestFrequencyDistributions(unittest.TestCase):
    @points(8)
    def test_bigram_frequency_dist(self) -> None:
        dist = bigram_freq_dist(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt"))
        )
        self.assertEqual(2, dist["The"]["cat"])
        self.assertEqual(3, dist["The"]["dog"])
        self.assertEqual(1, dist["the"]["cat"])
        self.assertEqual(7, dist["."][END_TOKEN0])

    @points(2)
    def test_bigram_frequency_dist_lower(self) -> None:
        dist = bigram_freq_dist(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt")),
            lower=True,
        )
        self.assertEqual(0, dist["The"]["cat"])
        self.assertEqual(3, dist["the"]["dog"])
        self.assertEqual(3, dist["the"]["cat"])
        self.assertEqual(7, dist["."][END_TOKEN0])

    @points(8)
    def test_trigram_frequency_dist(self) -> None:
        dist = trigram_freq_dist(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt"))
        )
        self.assertEqual(2, dist[("The", "dog")]["drank"])
        self.assertEqual(1, dist[("squirrel", "ate")]["peanuts"])
        self.assertEqual(7, dist[(".", END_TOKEN0)][END_TOKEN1])
        self.assertEqual(7, dist[(START_TOKEN1, START_TOKEN0)]["The"])
        self.assertEqual(3, sum(dist[("The", "dog")].values()))

    @points(2)
    def test_trigram_frequency_dist_lower(self) -> None:
        dist = trigram_freq_dist(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt")),
            lower=True,
        )
        self.assertEqual(0, dist[("The", "cat")]["ate"])
        self.assertEqual(2, dist[("the", "dog")]["drank"])
        self.assertEqual(1, dist[("the", "cat")]["ate"])
        self.assertEqual(3, sum(dist[("the", "cat")].values()))


class TestProbabilities(unittest.TestCase):
    @points(1)
    def test_unigram_probabilities_type(self) -> None:
        probs = unigram_probabilities(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt")),
            lower=True,
        )
        self.assertEqual(defaultdict, type(probs))

    @points(1)
    def test_bigram_probabilities_type(self) -> None:
        probs = bigram_probabilities(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt")),
            lower=True,
        )
        self.assertEqual(defaultdict, type(probs))

    @points(1)
    def test_trigram_probabilities_type(self) -> None:
        probs = trigram_probabilities(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt")),
            lower=True,
        )
        self.assertEqual(defaultdict, type(probs))

    @points(4)
    def test_unigram_probabilities(self) -> None:
        probs = unigram_probabilities(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt")),
            lower=True,
        )
        self.assertAlmostEqual(0.26315789, probs["the"])
        self.assertAlmostEqual(0.07894736, probs["dog"])
        self.assertAlmostEqual(0.02631578, probs["pizza"])
        self.assertEqual(0, probs["cookies"])

    @points(5)
    def test_bigram_probabilities(self) -> None:
        probs = bigram_probabilities(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt")),
            lower=True,
        )
        self.assertAlmostEqual(0.3, probs["the"]["dog"])
        self.assertEqual(
            1, probs[START_TOKEN0]["the"],
        )
        self.assertEqual(1, probs["squirrel"]["ate"])
        self.assertEqual(0, probs["cookies"]["are"])

    @points(5)
    def test_trigram_probabilities(self) -> None:
        probs = trigram_probabilities(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt")),
            lower=True,
        )
        self.assertAlmostEqual(0.5, probs[("dog", "drank")]["coffee"])
        self.assertAlmostEqual(
            0.4285714285, probs[(START_TOKEN0, "the")]["dog"],
        )
        self.assertAlmostEqual(0.5, probs[("squirrel", "ate")]["peanuts"])
        self.assertEqual(0, probs[("cookies", "are")]["good"])


class TestComparison(unittest.TestCase):
    @points(3)
    def compare_unigram_probabilities(self) -> None:
        diff = unigram_probabilities_difference(
            unigram_probabilities(
                gen_sentences(os.path.join("test_data", "ladygaga.txt")), lower=True
            ),
            unigram_probabilities(
                gen_sentences(os.path.join("test_data", "KingJames.txt")), lower=True
            ),
            intersection_only=False,
        )
        self.assertAlmostEqual(0.01179526361706696, diff["#"])
        self.assertAlmostEqual(0.011647311103199175, diff["the"])
        self.assertAlmostEqual(0.008741258741258742, diff["chromatica"])

    @points(3)
    def compare_unigram_probabilities_intersection(self) -> None:
        diff = unigram_probabilities_difference(
            unigram_probabilities(
                gen_sentences(os.path.join("test_data", "ladygaga.txt")), lower=True
            ),
            unigram_probabilities(
                gen_sentences(os.path.join("test_data", "KingJames.txt")), lower=True
            ),
            intersection_only=True,
        )
        self.assertAlmostEqual(0.01179526361706696, diff["#"])
        self.assertAlmostEqual(0.011647311103199175, diff["the"])
        self.assertAlmostEqual(0.005001338537094464, diff["love"])

    @points(3)
    def compare_bigram_probabilities(self) -> None:
        bigram_diff = bigram_probabilities_difference(
            bigram_probabilities(
                gen_sentences(os.path.join("test_data", "ladygaga.txt")), lower=True
            ),
            bigram_probabilities(
                gen_sentences(os.path.join("test_data", "lizzo.txt")), lower=True
            ),
            "i",
            intersection_only=False,
        )
        self.assertAlmostEqual(0.08695652173913043, bigram_diff["want"])
        self.assertAlmostEqual(0.043478260869565216, bigram_diff["wish"])

    @points(3)
    def compare_bigram_probabilities_intersection(self) -> None:
        bigram_diff = bigram_probabilities_difference(
            bigram_probabilities(
                gen_sentences(os.path.join("test_data", "lizzo.txt")), lower=True
            ),
            bigram_probabilities(
                gen_sentences(os.path.join("test_data", "ladygaga.txt")), lower=True
            ),
            "i",
            intersection_only=True,
        )
        self.assertAlmostEqual(0.014624505928853754, bigram_diff["hope"])
        self.assertAlmostEqual(0.011462450592885365, bigram_diff["love"])


def main() -> None:
    tests = [
        TestGenSentences,
        TestNGrams,
        TestCounts,
        TestFrequencyDistributions,
        TestProbabilities,
        TestComparison,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
