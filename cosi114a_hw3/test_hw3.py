#! /usr/bin/env python

# test_hw3.py
# Version 1.0
# 10/6/2020

import os
import unittest
from typing import Tuple, List

from grader import Grader, points
from hw3 import (
    accuracy,
    precision,
    recall,
    f1,
    load_segmentation_instances,
    ClassificationInstance,
    load_airline_instances,
    InstanceCounter,
    SentenceSplitInstance,
    NaiveBayesClassifier,
    BaselineAirlineSentimentFeatureExtractor,
    BaselineSegmentationFeatureExtractor,
    AirlineSentimentInstance,
    TunedSegmentationFeatureExtractor,
)

SENTENCE_SPLIT_DIR = os.path.join("test_data", "sentence_splits")
AIRLINE_SENTIMENT_DIR = os.path.join("test_data", "airline_sentiment")


class TestScoringMetrics(unittest.TestCase):
    @points(3)
    def test_accuracy(self):
        """Test accuracy function"""
        predicted = ["T", "T", "T", "T", "F", "F", "F", "F", "T", "T"]
        actual = ["T", "T", "T", "T", "T", "F", "F", "F", "F", "F"]
        self.assertEqual(float, type(accuracy(predicted, actual)))
        self.assertAlmostEqual(0.7, accuracy(predicted, actual))

    @points(3)
    def test_precision(self):
        """Test precision function"""
        predicted = ["T", "T", "T", "T", "F", "F", "F", "F", "T", "T"]
        actual = ["T", "T", "T", "T", "T", "F", "F", "F", "F", "F"]
        self.assertEqual(float, type(precision(predicted, actual, "T")))
        self.assertAlmostEqual(2 / 3, precision(predicted, actual, "T"))
        self.assertAlmostEqual(0.75, precision(predicted, actual, "F"))

    @points(3)
    def test_recall(self):
        """Test recall function"""
        predicted = ["T", "T", "T", "T", "F", "F", "F", "F", "T", "T"]
        actual = ["T", "T", "T", "T", "T", "F", "F", "F", "F", "F"]
        self.assertEqual(float, type(recall(predicted, actual, "T")))
        self.assertAlmostEqual(0.8, recall(predicted, actual, "T"))
        self.assertAlmostEqual(3 / 5, recall(predicted, actual, "F"))

    @points(3)
    def test_f1_score(self):
        """Test F1 scoring function"""
        predicted = ["T", "T", "T", "T", "F", "F", "F", "F", "T", "T"]
        actual = ["T", "T", "T", "T", "T", "F", "F", "F", "F", "F"]
        self.assertEqual(float, type(f1(predicted, actual, "T")))
        self.assertAlmostEqual(8 / 11, f1(predicted, actual, "T"))
        self.assertAlmostEqual(2 / 3, f1(predicted, actual, "F"))


class TestFeatureExtractor(unittest.TestCase):
    @points(3)
    def test_sentence_split_feature_extractor(self):
        """Test feature extractor for sentence segmentation"""
        label_set = frozenset(["y", "n"])
        seg_feature_extractor = BaselineSegmentationFeatureExtractor()
        instances = load_segmentation_instances(
            os.path.join(SENTENCE_SPLIT_DIR, "dev.json")
        )
        classification_instance = seg_feature_extractor.extract_features(
            next(instances)
        )
        self.assertEqual(ClassificationInstance, type(classification_instance))
        self.assertEqual(list, type(classification_instance.features))
        self.assertGreaterEqual(3, len(classification_instance.features))
        self.assertEqual(str, type(classification_instance.features[0]))
        self.assertEqual("n", classification_instance.label)
        self.assertSetEqual(
            {"split_tok=.", "left_tok=D", "right_tok=Forrester"},
            set(classification_instance.features),
        )

        for inst in instances:
            classify_inst = seg_feature_extractor.extract_features(inst)
            self.assertIn(classify_inst.label, label_set)

    @points(3)
    def test_airline_sentiment_feature_extractor(self):
        """Test feature extractor for sentiment detection of airline tweets"""
        label_set = frozenset(["positive", "negative"])
        air_feat_extractor = BaselineAirlineSentimentFeatureExtractor()
        instances = load_airline_instances(
            os.path.join(AIRLINE_SENTIMENT_DIR, "dev.json")
        )
        classification_instance = air_feat_extractor.extract_features(next(instances))
        self.assertEqual(ClassificationInstance, type(classification_instance))
        self.assertEqual(list, type(classification_instance.features))
        self.assertEqual(str, type(classification_instance.features[0]))
        self.assertEqual("negative", classification_instance.label)
        self.assertSetEqual(
            {
                "several",
                "&",
                "route",
                "take",
                "be",
                "+",
                "that",
                "self",
                ",",
                "@united",
                "year",
                "we",
                "times",
                "amp",
                "would",
                "situation",
                "hate",
                "a",
                "interest",
                ".",
                "to",
                ";",
                "degree",
                "-",
                "of",
                "in",
                "besides",
                "no",
                "small",
            },
            set(classification_instance.features),
        )

        for inst in instances:
            classify_inst = air_feat_extractor.extract_features(inst)
            self.assertIn(classify_inst.label, label_set)


class SegmentationTestFeatureExtractor:
    """
    Simple baseline feature extractor to test InstanceCounter and NaiveBayes independently of what
    the real feature extractors are choosing.
    """

    def extract_features(self, inst: SentenceSplitInstance) -> ClassificationInstance:
        return ClassificationInstance(
            inst.label, [f"left_tok={inst.left_context}", f"split_tok={inst.token}"]
        )


class SentimentTestFeatureExtractor:
    """
    Simple baseline feature extractor to test InstanceCounter and NaiveBayes independently of what
    the real feature extractors are choosing.
    """

    def __init__(self):
        self.words = frozenset(["thank", "bad", "great", "good", "like", "you"])

    def extract_features(
        self, inst: AirlineSentimentInstance
    ) -> ClassificationInstance:
        return ClassificationInstance(
            inst.label,
            [
                tok
                for sent in inst.sentences
                for tok in sent
                if tok.lower() in self.words
            ],
        )


class TestInstanceCounter(unittest.TestCase):
    def setUp(self) -> None:
        # Create instance counter and count the instances
        self.inst_counter = InstanceCounter()
        feature_extractor = SegmentationTestFeatureExtractor()
        self.inst_counter.count_instances(
            feature_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "dev.json")
            )
        )

    @points(2)
    def test_label_counts_y(self):
        self.assertEqual(6110, self.inst_counter.label_count("y"))

    @points(2)
    def test_label_counts_n(self):
        self.assertEqual(811, self.inst_counter.label_count("n"))

    @points(1)
    def test_total_labels(self):
        self.assertEqual(6921, self.inst_counter.total_labels())

    @points(2)
    def test_conditional_feature_count_1(self):
        self.assertEqual(
            5903, self.inst_counter.conditional_feature_count("y", "split_tok=.")
        )

    @points(2)
    def test_conditional_feature_count_2(self):
        self.assertEqual(
            751, self.inst_counter.conditional_feature_count("n", "split_tok=.")
        )

    @points(4)
    def test_labels(self):
        labels = frozenset(["y", "n"])
        for label in self.inst_counter.labels():
            self.assertIn(label, labels)
        self.assertEqual(2, len(self.inst_counter.labels()))

    @points(3)
    def test_feature_vocab_size(self):
        self.assertEqual(2964, self.inst_counter.feature_vocab_size())

    @points(3)
    def test_total_feature_count_for_class(self):
        self.assertEqual(12220, self.inst_counter.total_feature_count_for_class("y"))
        self.assertEqual(1622, self.inst_counter.total_feature_count_for_class("n"))


class TestNaiveBayesSegmentation(unittest.TestCase):
    def setUp(self):
        """Load data and train classifiers"""
        # Setup and train segmentation classifier
        segmentation_extractor = SegmentationTestFeatureExtractor()
        segmentation_train_instances = (
            segmentation_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "train.json")
            )
        )
        self.segmentation_dev_instances = [
            segmentation_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "dev.json")
            )
        ]
        self.segmentation_classifier = NaiveBayesClassifier(k=0.01)
        self.segmentation_classifier.train(segmentation_train_instances)

    @points(3)
    def test_prior_probability(self):
        "Test prior probability in log space"
        self.assertAlmostEqual(
            -0.14306728814197256, self.segmentation_classifier.prior_prob("y")
        )
        self.assertAlmostEqual(
            -2.0151211587336735, self.segmentation_classifier.prior_prob("n")
        )

    @points(3)
    def test_likelihood_prob(self):
        """Test likelihood_prob method in log space"""
        self.assertAlmostEqual(
            -0.7236102318647165,
            self.segmentation_classifier.likelihood_prob("split_tok=.", "y"),
        )
        self.assertAlmostEqual(
            -5.655509526487457,
            self.segmentation_classifier.likelihood_prob("split_tok=!", "y"),
        )
        self.assertAlmostEqual(
            -0.7336498797029166,
            self.segmentation_classifier.likelihood_prob("split_tok=.", "n"),
        )
        self.assertAlmostEqual(
            -5.1237691773702085,
            self.segmentation_classifier.likelihood_prob("split_tok=!", "n"),
        )

    @points(3)
    def test_probability_segmentation(self):
        self.assertAlmostEqual(
            -0.8666775200066891, self.segmentation_classifier.prob(["split_tok=."], "y")
        )
        self.assertAlmostEqual(
            -5.798576814629429, self.segmentation_classifier.prob(["split_tok=!"], "y")
        )
        self.assertAlmostEqual(
            -2.74877103843659, self.segmentation_classifier.prob(["split_tok=."], "n")
        )
        self.assertAlmostEqual(
            -7.1388903361038825, self.segmentation_classifier.prob(["split_tok=!"], "n")
        )

    @points(2)
    def test_classify(self):
        self.assertEqual(
            "y",
            self.segmentation_classifier.classify(["left_tok=products", "split_tok=."]),
        )
        self.assertEqual(
            "n", self.segmentation_classifier.classify(["left_tok=Dr", "split_tok=."])
        )
        self.assertEqual(
            str, type(self.segmentation_classifier.classify(["split_tok=."]))
        )

    @points(3)
    def test_naivebayes_test(self):
        result = self.segmentation_classifier.test(
            [
                ClassificationInstance("y", ["left_tok=outstanding", "split_tok=."]),
                ClassificationInstance("y", ["left_tok=fairly", "split_tok=?"]),
                ClassificationInstance("n", ["left_tok=U.S", "split_tok=."]),
                ClassificationInstance("y", ["left_tok=!", "split_tok=!"]),
                ClassificationInstance("n", ["left_tok=Mx.", "split_tok=."]),
            ]
        )
        self.assertEqual(tuple, type(result))
        self.assertEqual(list, type(result[0]))
        self.assertEqual(list, type(result[1]))
        self.assertEqual(len(result[0]), len(result[1]))
        for item in result[0]:
            self.assertEqual(str, type(item))
        for item in result[1]:
            self.assertEqual(str, type(item))
        self.assertEqual((["y", "y", "n", "y", "y"], ["y", "y", "n", "y", "n"]), result)


class TestNaiveBayesSentiment(unittest.TestCase):
    def setUp(self):
        """Load data and train classifier"""
        airline_extractor = SentimentTestFeatureExtractor()
        airline_train_instances = (
            airline_extractor.extract_features(inst)
            for inst in load_airline_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "train.json")
            )
        )
        self.airline_dev_instances = [
            airline_extractor.extract_features(inst)
            for inst in load_airline_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "dev.json")
            )
        ]
        self.airline_classifier = NaiveBayesClassifier(0.01)
        self.airline_classifier.train(airline_train_instances)

    @points(3)
    def test_prior_probability(self):
        "Test prior probability in log space"
        self.assertAlmostEqual(
            -1.5881551483595602, self.airline_classifier.prior_prob("positive")
        )
        self.assertAlmostEqual(
            -0.22853577761478316, self.airline_classifier.prior_prob("negative")
        )

    @points(3)
    def test_likelihood_prob(self):
        """Test likelihood_prob method in log space"""
        self.assertAlmostEqual(
            -2.1908333383500493,
            self.airline_classifier.likelihood_prob("thank", "positive"),
        )
        self.assertAlmostEqual(
            -4.704411374296565,
            self.airline_classifier.likelihood_prob("bad", "positive"),
        )
        self.assertAlmostEqual(
            -4.491450754855788,
            self.airline_classifier.likelihood_prob("thank", "negative"),
        )
        self.assertAlmostEqual(
            -3.2968216795806224,
            self.airline_classifier.likelihood_prob("bad", "negative"),
        )

    @points(3)
    def test_probability(self):
        self.assertAlmostEqual(
            -3.7789884867096095, self.airline_classifier.prob(["thank"], "positive")
        )
        self.assertAlmostEqual(
            -6.292566522656125, self.airline_classifier.prob(["bad"], "positive")
        )
        self.assertAlmostEqual(
            -4.719986532470571, self.airline_classifier.prob(["thank"], "negative")
        )
        self.assertAlmostEqual(
            -3.5253574571954056, self.airline_classifier.prob(["bad"], "negative")
        )

    @points(2)
    def test_classify(self):
        self.assertEqual("positive", self.airline_classifier.classify(["thank"]))
        self.assertEqual("negative", self.airline_classifier.classify(["bad"]))
        self.assertEqual(str, type(self.airline_classifier.classify(["thank"])))


class TestPerformanceSegmentation(unittest.TestCase):
    def setUp(self):
        """Load data and train classifiers"""
        segmentation_extractor = BaselineSegmentationFeatureExtractor()
        segmentation_train_instances = (
            segmentation_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "train.json")
            )
        )
        self.segmentation_dev_instances = [
            segmentation_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "dev.json")
            )
        ]
        self.segmentation_classifier = NaiveBayesClassifier(2.0)
        self.segmentation_classifier.train(segmentation_train_instances)

    @points(4)
    def test_segmentation_performance_y(self):
        predicted, expected = self.segmentation_classifier.test(
            self.segmentation_dev_instances
        )
        acc, prec, rec, f1_score, report = classification_report(
            predicted, expected, "y"
        )
        print("Baseline segmentation performance:")
        print(report)

        self.assertLessEqual(0.986, acc)
        self.assertLessEqual(0.987, prec)
        self.assertLessEqual(0.997, rec)
        self.assertLessEqual(0.992, f1_score)


class TestPerformanceSentiment(unittest.TestCase):
    def setUp(self):
        """Load data and train classifiers"""
        airline_extractor = BaselineAirlineSentimentFeatureExtractor()
        airline_train_instances = (
            airline_extractor.extract_features(inst)
            for inst in load_airline_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "train.json")
            )
        )
        self.airline_dev_instances = [
            airline_extractor.extract_features(inst)
            for inst in load_airline_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "dev.json")
            )
        ]
        self.airline_classifier = NaiveBayesClassifier(0.05)
        self.airline_classifier.train(airline_train_instances)

    @points(2)
    def test_sentiment_performance_positive(self):
        predicted, expected = self.airline_classifier.test(self.airline_dev_instances)
        acc, prec, rec, f1_score, report = classification_report(
            predicted, expected, "positive"
        )
        print("Baseline positive sentiment performance:")
        print(report)

        self.assertLessEqual(0.896, acc)
        self.assertLessEqual(0.741, prec)
        self.assertLessEqual(0.798, rec)
        self.assertLessEqual(0.768, f1_score)

    @points(2)
    def test_sentiment_performance_negative(self):
        predicted, expected = self.airline_classifier.test(self.airline_dev_instances)
        acc, prec, rec, f1_score, report = classification_report(
            predicted, expected, "negative"
        )
        print("Baseline negative sentiment performance:")
        print(report)

        self.assertLessEqual(0.896, acc)
        self.assertLessEqual(0.943, prec)
        self.assertLessEqual(0.923, rec)
        self.assertLessEqual(0.933, f1_score)


class TestTunedSegmentation(unittest.TestCase):
    @points(0)
    def test_tuned_segmentation(self):
        segmentation_extractor = TunedSegmentationFeatureExtractor()
        self.assertIsNotNone(segmentation_extractor.k)

        segmentation_train_instances = (
            segmentation_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "train.json")
            )
        )
        self.segmentation_dev_instances = [
            segmentation_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "dev.json")
            )
        ]
        self.segmentation_classifier = NaiveBayesClassifier(segmentation_extractor.k)
        self.segmentation_classifier.train(segmentation_train_instances)
        predicted, expected = self.segmentation_classifier.test(
            self.segmentation_dev_instances
        )
        acc, prec, rec, f1_score, report = classification_report(
            predicted, expected, "y"
        )
        print(f"Tuned segmentation performance for k of {segmentation_extractor.k}:")
        print(report)


def classification_report(
    predicted: List[str], expected: List[str], positive_label: str,
) -> Tuple[float, float, float, float, str]:
    """Return accuracy, P, R, F1 and a classification report."""
    acc = accuracy(predicted, expected)
    prec = precision(predicted, expected, positive_label)
    rec = recall(predicted, expected, positive_label)
    f1_score = f1(predicted, expected, positive_label)
    report = "\n".join(
        [
            f"Accuracy:  {acc * 100:0.2f}",
            f"Precision: {prec * 100:0.2f}",
            f"Recall:    {rec * 100:0.2f}",
            f"F1:        {f1_score * 100:0.2f}",
        ]
    )
    return acc, prec, rec, f1_score, report


def main() -> None:
    tests = [
        TestScoringMetrics,
        TestFeatureExtractor,
        TestInstanceCounter,
        TestNaiveBayesSegmentation,
        TestNaiveBayesSentiment,
        TestPerformanceSegmentation,
        TestPerformanceSentiment,
        TestTunedSegmentation,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
