#! /usr/bin/env python

# test_hw4.py
# Version 1.2
# 11/01/2020

import random
import unittest
from typing import Tuple, List, Generator

from grader import Grader, points
from hw4 import (
    MostFrequentTagTagger,
    SentenceCounter,
    Token,
    UnigramTagger,
    GreedyBigramTagger,
    ViterbiBigramTagger,
)

from sklearn.metrics import accuracy_score


def make_sentences(sentences: List[List[Tuple[str, str]]]) -> List[List[Token]]:
    return [[Token.from_tuple(pair) for pair in sentence] for sentence in sentences]


# Has to be defined below make_sentences
SENTENCES_AB_XYZ = make_sentences(
    [
        [("x", "A"), ("x", "A"), ("y", "A"), ("z", "A"), ("z", "A")],
        [("x", "B"), ("y", "B"), ("y", "B"), ("y", "B"), ("z", "B")],
    ]
)

# many different tags so odds of accidentally getting right first tag is lower
SENTENCES_NN_VB_XZY = make_sentences(
    [
        [("pumpkin", "VBD"), ("ghost", "VBD"), ("cider", "VBD")],
        [("pumpkin", "VB"), ("ghost", "VB"), ("cider", "VB")],
        [("pumpkin", "VBG"), ("ghost", "VBG"), ("cider", "VBG")],
        [("pumpkin", "NN"), ("ghost", "NN"), ("cider", "NN")],
        [("pumpkin", "#"), ("ghost", "#"), ("cider", "#")],
        [("pumpkin", "DT"), ("ghost", "DT"), ("cider", "DT")],
        [("pumpkin", "."), ("ghost", "."), ("cider", ".")],
        [("pumpkin", "NNP"), ("ghost", "NNP"), ("cider", "NNP")],
    ]
)


def load_pos_data(path):
    with open(path, "r", encoding="utf8") as infile:
        for line in infile:
            if not line.strip():
                continue
            yield [Token.from_string(tok) for tok in line.rstrip("\n").split(" ")]


class TestMostFrequentTagTagger(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences = make_sentences(
            [[("foo", "NN"), ("bar", "NNS")], [("baz", "NN")]]
        )
        self.tagger = MostFrequentTagTagger()
        self.tagger.train(self.sentences)

    @points(1)
    def test_most_frequent_tag_sentence(self):
        sentence = "This is a sentence .".split(" ")
        self.assertListEqual(
            ["NN", "NN", "NN", "NN", "NN"], self.tagger.tag_sentence(sentence)
        )

    @points(1)
    def test_most_freq_accuracy(self):
        self.tagger.train(load_pos_data("test_data/train_pos.txt"))
        tagged_sents = load_pos_data("test_data/100_dev.txt")
        predicted, actual = self.tagger.test(tagged_sents)
        accuracy = accuracy_score(actual, predicted)
        print(f"MostFrequentTagTagger Accuracy: {accuracy * 100:.2f}")
        self.assertLessEqual(0.128, accuracy)


class TestUnigramTagger(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences = make_sentences(
            [
                [("foo", "NN"), ("foo", "NNS"), ("bar", "JJ")],
                [("foo", "NN"), ("bar", "JJ")],
                [("baz", "VB")],
            ]
        )
        self.tagger = UnigramTagger()
        self.tagger.train(self.sentences)

    @points(1)
    def test_tag_foo(self):
        tags = self.tagger.tag_sentence(["foo"])
        self.assertEqual("NN", tags[0])

    @points(1)
    def test_tag_bar(self):
        tags = self.tagger.tag_sentence(["bar"])
        self.assertEqual("JJ", tags[0])

    @points(1)
    def test_tag_baz(self):
        tags = self.tagger.tag_sentence(["baz"])
        self.assertEqual("VB", tags[0])

    @points(2)
    def test_unigram_tagger_accuracy(self):
        self.tagger.train(load_pos_data("test_data/train_pos.txt"))
        tagged_sents = load_pos_data("test_data/100_dev.txt")
        predicted, actual = self.tagger.test(tagged_sents)
        accuracy = accuracy_score(actual, predicted)
        print(f"UnigramTagger Accuracy: {accuracy * 100:.2f}")
        self.assertLessEqual(0.929, accuracy)


class TestInstanceCounterUnsmoothed(unittest.TestCase):
    def setUp(self):
        self.inst_counter = SentenceCounter(0.0)
        sentences = make_sentences(
            [
                [("e", "C"), ("f", "A"), ("e", "C")],
                [("h", "B"), ("f", "C"), ("g", "B")],
            ]
        )
        self.inst_counter.count_sentences(sentences)

    @points(1.5)
    def test_tagset(self):
        self.assertEqual(["A", "B", "C"], self.inst_counter.tagset())

    @points(1)
    def test_emission_prob1(self):
        self.assertAlmostEqual(2 / 3, self.inst_counter.emission_prob("C", "e"))
        self.assertAlmostEqual(1 / 3, self.inst_counter.emission_prob("C", "f"))

    @points(1)
    def test_emission_prob2(self):
        self.assertAlmostEqual(0.5, self.inst_counter.emission_prob("B", "g"))
        self.assertAlmostEqual(0.5, self.inst_counter.emission_prob("B", "h"))

    @points(1)
    def test_emission_prob3(self):
        self.assertEqual(1.0, self.inst_counter.emission_prob("A", "f"))

    @points(1)
    def test_emission_prob4(self):
        self.assertEqual(0.0, self.inst_counter.emission_prob("A", "e"))
        self.assertEqual(0.0, self.inst_counter.emission_prob("B", "f"))
        self.assertEqual(0.0, self.inst_counter.emission_prob("C", "g"))

    @points(1)
    def test_intial_prob1(self):
        self.assertEqual(0.5, self.inst_counter.initial_prob("C"))
        self.assertEqual(0.5, self.inst_counter.initial_prob("B"))

    @points(1)
    def test_intial_prob2(self):
        self.assertEqual(0.0, self.inst_counter.initial_prob("A"))

    @points(1)
    def test_transition_prob1(self):
        self.assertEqual(0.5, self.inst_counter.transition_prob("C", "A"))
        self.assertEqual(0.5, self.inst_counter.transition_prob("C", "B"))

    @points(1)
    def test_transition_prob2(self):
        self.assertEqual(1.0, self.inst_counter.transition_prob("A", "C"))
        self.assertEqual(1.0, self.inst_counter.transition_prob("B", "C"))

    @points(1)
    def test_transition_prob3(self):
        self.assertEqual(0.0, self.inst_counter.transition_prob("A", "B"))
        self.assertEqual(0.0, self.inst_counter.transition_prob("B", "A"))


class TestInstanceCounterSmoothed(unittest.TestCase):
    def setUp(self):
        self.inst_counter = SentenceCounter(1.0)
        sentences = make_sentences(
            [
                [("e", "C"), ("f", "A"), ("e", "C")],
                [("h", "B"), ("f", "C"), ("g", "B")],
            ]
        )
        self.inst_counter.count_sentences(sentences)

    @points(1)
    def test_emission_prob1(self):
        self.assertAlmostEqual(3 / 5, self.inst_counter.emission_prob("C", "e"))
        self.assertAlmostEqual(2 / 5, self.inst_counter.emission_prob("C", "f"))

    @points(1)
    def test_emission_prob2(self):
        self.assertAlmostEqual(2 / 4, self.inst_counter.emission_prob("B", "g"))
        self.assertAlmostEqual(2 / 4, self.inst_counter.emission_prob("B", "h"))

    @points(1)
    def test_emission_prob3(self):
        self.assertEqual(1.0, self.inst_counter.emission_prob("A", "f"))

    @points(1)
    def test_emission_prob4(self):
        self.assertEqual(1 / 2, self.inst_counter.emission_prob("A", "e"))
        self.assertEqual(1 / 4, self.inst_counter.emission_prob("B", "f"))
        self.assertEqual(1 / 5, self.inst_counter.emission_prob("C", "g"))

    # Initial/transition probabilities are not affected by smoothing, so these tests
    # give minimal points
    @points(0.1)
    def test_intial_prob1(self):
        self.assertEqual(0.5, self.inst_counter.initial_prob("C"))
        self.assertEqual(0.5, self.inst_counter.initial_prob("B"))

    @points(0.1)
    def test_intial_prob2(self):
        self.assertEqual(0.0, self.inst_counter.initial_prob("A"))

    @points(0.1)
    def test_transition_prob1(self):
        self.assertEqual(0.5, self.inst_counter.transition_prob("C", "A"))
        self.assertEqual(0.5, self.inst_counter.transition_prob("C", "B"))

    @points(0.1)
    def test_transition_prob2(self):
        self.assertEqual(1.0, self.inst_counter.transition_prob("A", "C"))
        self.assertEqual(1.0, self.inst_counter.transition_prob("B", "C"))

    @points(0.1)
    def test_transition_prob3(self):
        self.assertEqual(0.0, self.inst_counter.transition_prob("A", "B"))
        self.assertEqual(0.0, self.inst_counter.transition_prob("B", "A"))


class TestSentenceCounterSpeed(unittest.TestCase):
    @points(5)
    def test_efficient_implementation(self):
        """Test that SentenceCounter is efficiently implemented.

        If you are failing this test with a TimeoutError, you are not implementing
        SentenceCounter efficiently and are probably using loops or sums in your
        probability functions. Instead, precompute values in count_sentences.
        """
        counter = SentenceCounter(1.0)
        counter.count_sentences(self._make_random_sentences(75_000))
        for _ in range(1000):
            counter.tagset()
            counter.initial_prob("A")
            counter.transition_prob("A", "A")
            counter.emission_prob("A", "1")

    @staticmethod
    def _make_random_sentences(n_sentences: int) -> Generator[List[Token], None, None]:
        random.seed(0)
        tags = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        tokens = [str(n) for n in range(n_sentences)]
        lengths = list(range(10, 31))
        for _ in range(n_sentences):
            sen_length = random.choice(lengths)
            sentence = [
                Token(random.choice(tokens), random.choice(tags))
                for _ in range(sen_length)
            ]
            yield sentence


class TestBigramSequenceProbability(unittest.TestCase):
    def setUp(self):
        # We test through the greedy tagger but could also do it through Viterbi
        self.tagger = GreedyBigramTagger(0.0)
        self.tagger.train(SENTENCES_AB_XYZ)

    @points(2)
    def test_prob1(self):
        self.assertAlmostEqual(
            -3.2188758248682006,
            self.tagger.sequence_probability(["x", "y"], ["A", "A"]),
        )

    @points(1)
    def test_prob2(self):
        self.assertEqual(
            float("-inf"), self.tagger.sequence_probability(["x", "y"], ["A", "B"])
        )

    @points(1)
    def test_prob3(self):
        self.assertEqual(
            float("-inf"), self.tagger.sequence_probability(["x", "y"], ["B", "A"])
        )

    @points(2)
    def test_prob4(self):
        self.assertAlmostEqual(
            -2.8134107167600364,
            self.tagger.sequence_probability(["x", "y"], ["B", "B"]),
        )


class TestGreedyBigramTagger(unittest.TestCase):
    def setUp(self):
        self.tagger = GreedyBigramTagger(0.0)
        self.tagger.train(SENTENCES_AB_XYZ)

    @points(2)
    def test_ab_xyz_tag1(self):
        sent = ["x", "x"]
        self.assertEqual(["A", "A"], self.tagger.tag_sentence(sent))

    @points(2)
    def test_ab_xyz_tag2(self):
        sent = ["y", "y"]
        self.assertEqual(["B", "B"], self.tagger.tag_sentence(sent))

    @points(2)
    def test_ab_xyz_tag4(self):
        sent = ["x", "y"]
        self.assertEqual(["A", "A"], self.tagger.tag_sentence(sent))

    @points(2)
    def test_ab_xyz_tag3(self):
        sent = ["x", "z"]
        self.assertEqual(["A", "A"], self.tagger.tag_sentence(sent))

    @points(3)
    def test_greedy_tagger_accuracy(self):
        self.tagger = GreedyBigramTagger(0.001)
        self.tagger.train(load_pos_data("test_data/train_pos.txt"))
        tagged_sents = load_pos_data("test_data/100_dev.txt")
        predicted, actual = self.tagger.test(tagged_sents)
        accuracy = accuracy_score(actual, predicted)
        print(f"GreedyBigram Accuracy: {accuracy * 100:.2f}")
        self.assertLessEqual(0.953, accuracy)


class TestViterbiBigramTagger(unittest.TestCase):
    def setUp(self):
        self.tagger = ViterbiBigramTagger(0.0)
        self.tagger.train(SENTENCES_AB_XYZ)

    @points(4)
    def test_ab_xyz_tag1(self):
        sent = ["x", "x"]
        self.assertEqual(["A", "A"], self.tagger.tag_sentence(sent))

    @points(4)
    def test_ab_xyz_tag2(self):
        sent = ["y", "y"]
        self.assertEqual(["B", "B"], self.tagger.tag_sentence(sent))

    @points(4)
    def test_ab_xyz_tag4(self):
        sent = ["x", "y"]
        self.assertEqual(["B", "B"], self.tagger.tag_sentence(sent))

    @points(4)
    def test_ab_xyz_tag3(self):
        sent = ["x", "z"]
        self.assertEqual(["A", "A"], self.tagger.tag_sentence(sent))

    @points(6)
    def test_viterbi_tagger_accuracy(self):
        self.tagger = ViterbiBigramTagger(0.001)
        self.tagger.train(load_pos_data("test_data/train_pos.txt"))
        # Test on smaller dev set for speed purposes
        tagged_sents = load_pos_data("test_data/100_dev.txt")
        predicted, actual = self.tagger.test(tagged_sents)
        accuracy = accuracy_score(actual, predicted)
        print(f"ViterbiBigram Accuracy: {accuracy * 100:.2f}")
        self.assertLessEqual(0.962, accuracy)


class TestDeterminism(unittest.TestCase):
    def setUp(self) -> None:
        self.greedy_tagger = GreedyBigramTagger(0.0)
        self.viterbi_tagger = ViterbiBigramTagger(0.0)
        self.greedy_tagger.train(SENTENCES_NN_VB_XZY)
        self.viterbi_tagger.train(SENTENCES_NN_VB_XZY)

    @points(2)
    def test_greedy_determinism(self):
        sent = ["ghost", "pumpkin", "cider"]
        self.assertEqual(["#", "#", "#"], self.greedy_tagger.tag_sentence(sent))

    @points(2)
    def test_viterbi_determinism(self):
        sent = ["ghost", "pumpkin", "cider"]
        self.assertEqual(["#", "#", "#"], self.viterbi_tagger.tag_sentence(sent))


def main() -> None:
    tests = [
        TestMostFrequentTagTagger,
        TestUnigramTagger,
        TestInstanceCounterUnsmoothed,
        TestInstanceCounterSmoothed,
        TestSentenceCounterSpeed,
        TestBigramSequenceProbability,
        TestGreedyBigramTagger,
        TestViterbiBigramTagger,
        TestDeterminism,
    ]
    grader = Grader(tests, timeout=12)
    grader.print_results()


if __name__ == "__main__":
    main()
