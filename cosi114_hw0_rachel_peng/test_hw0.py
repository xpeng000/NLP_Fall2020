#! /usr/bin/env python

import os
import unittest
from types import GeneratorType

from grader import Grader, points
from hw0 import case_sarcastically, detokenize, gen_sentences


class TestGenSentences(unittest.TestCase):
    @points(5)
    def test_type(self) -> None:
        """Test that a generator is returned."""
        gen = gen_sentences(os.path.join("test_data", "hw0_tokenized_text_1.txt"))
        self.assertEqual(type(gen), GeneratorType)

    @points(10)
    def test_basic(self) -> None:
        """Test reading a basic file."""
        gen = gen_sentences(os.path.join("test_data", "hw0_tokenized_text_1.txt"))
        self.assertEqual(
            next(gen), ["Tokenized", "text", "is", "easy", "to", "work", "with", "."]
        )
        self.assertEqual(
            next(gen), ["Writing", "a", "tokenizer", "is", "a", "pain", "."]
        )
        with self.assertRaises(StopIteration):
            next(gen)

    @points(10)
    def test_advanced(self) -> None:
        """Test reading a more complex file."""
        gen = gen_sentences(os.path.join("test_data", "hw0_tokenized_text_2.txt"))
        self.assertEqual(next(gen), ["Hello", ",", "world", "!"])
        # Between these sentences, there is a line in the file with a single space,
        # which should be skipped over.
        self.assertEqual(next(gen), ["This", "is", "a", "normal", "sentence", "."])
        self.assertEqual(
            next(gen),
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
        )
        with self.assertRaises(StopIteration):
            next(gen)


class TestDetokenize(unittest.TestCase):
    @points(10)
    def test_simple(self) -> None:
        """Test a simple sentence."""
        self.assertEqual(detokenize(["Hello", ",", "world", "!"]), "Hello, world!")

    @points(5)
    def test_quotes1(self) -> None:
        """Test how quotes are handled."""
        self.assertEqual(
            detokenize(
                [
                    '"',
                    "I",
                    "don't",
                    "know",
                    "what",
                    "NLP",
                    "is",
                    ",",
                    '"',
                    "he",
                    "said.",
                ]
            ),
            '"I don\'t know what NLP is," he said.',
        )
        self.assertEqual(
            detokenize(
                ['"', "Too", "much", "punctuation", "!", '"', "they", "exclaimed", "."]
            ),
            '"Too much punctuation!" they exclaimed.',
        )

    @points(5)
    def test_quotes2(self) -> None:
        """Test how quotes are handled."""
        self.assertEqual(
            detokenize(
                [
                    "She",
                    "said",
                    ",",
                    '"',
                    "I",
                    "don't",
                    "like",
                    "punctuation",
                    ",",
                    "do",
                    "you",
                    "?",
                    '"',
                ]
            ),
            'She said, "I don\'t like punctuation, do you?"',
        )

    @points(5)
    def test_strange_punc(self) -> None:
        """Test unusual punctuation."""
        self.assertEqual(
            detokenize(
                [
                    "Punctuation",
                    "can",
                    "surprise",
                    "you",
                    ";",
                    "no",
                    "one",
                    "expects",
                    "the",
                    "interrobang",
                    "‽",
                ]
            ),
            "Punctuation can surprise you; no one expects the interrobang‽",
        )

    @points(5)
    def test_em_dash(self) -> None:
        """Test em dash."""
        self.assertEqual(
            detokenize(
                [
                    "The",
                    "em",
                    "dash",
                    "—",
                    "one",
                    "of",
                    "my",
                    "favorite",
                    "characters",
                    "—",
                    "is",
                    "often",
                    "mistaken",
                    "for",
                    "the",
                    "en",
                    "dash",
                    ".",
                ]
            ),
            "The em dash—one of my favorite characters—is often mistaken for the en dash.",
        )


class TestSarcasticCaser(unittest.TestCase):
    @points(5)
    def test_no_punc(self) -> None:
        """Test basic text."""
        assert case_sarcastically("hello") == "hElLo"

    @points(10)
    def test_punc1(self) -> None:
        """Test how punctuation is handled."""
        assert case_sarcastically("hello, friend!") == "hElLo, FrIeNd!"

    @points(10)
    def test_punc2(self) -> None:
        """Test how punctuation is handled."""
        assert case_sarcastically('Say "hello," friend‽') == 'sAy "HeLlO," fRiEnD‽'


def main() -> None:
    tests = [
        TestGenSentences,
        TestDetokenize,
        TestSarcasticCaser,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
