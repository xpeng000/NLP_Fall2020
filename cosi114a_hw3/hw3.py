# test_hw3.py
# Version 1.0
# 10/6/2020

import json
import math
from collections import defaultdict, Counter
from math import log
from typing import (
    Iterable,
    Dict,
    Any,
    Tuple,
    List,
    Sequence,
    Generator,
)


############################################################
# Classes to represent instances of the data
class AirlineSentimentInstance:
    # DO NOT CHANGE THIS
    def __init__(self, label: str, sentences: List[List[str]], airline: str) -> None:
        self.label = label
        self.sentences = sentences
        self.airline = airline

    def __repr__(self):
        return f"label={self.label}; sentences={self.sentences}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "sentences": self.sentences,
            "airline": self.airline,
        }

    @classmethod
    def from_dict(cls, json_dict: Dict[str, Any]) -> "AirlineSentimentInstance":
        return AirlineSentimentInstance(
            json_dict["label"], json_dict["sentences"], json_dict["airline"]
        )


class SentenceSplitInstance:
    # DO NOT CHANGE THIS
    def __init__(self, label: str, left_context: str, token: str, right_context: str):
        self.label = label
        self.left_context = left_context
        self.token = token
        self.right_context = right_context

    def __repr__(self):
        return " ".join(
            [
                f"label={self.label};",
                f"left={self.left_context};",
                f"token={self.token};",
                f"right={self.right_context}",
            ]
        )

    def to_dict(self):
        return {
            "label": self.label,
            "left": self.left_context,
            "token": self.token,
            "right": self.right_context,
        }

    @classmethod
    def from_dict(cls, json_dict: Dict[Any, Any]):
        return SentenceSplitInstance(
            json_dict["label"],
            json_dict["left"],
            json_dict["token"],
            json_dict["right"],
        )


############################################################
# Functions to load instances
def load_airline_instances(
    datapath: str,
) -> Generator[AirlineSentimentInstance, None, None]:
    # DO NOT CHANGE THIS
    with open(datapath, "r", encoding="utf8") as infile:
        json_list = json.load(infile)
        for json_item in json_list:
            yield AirlineSentimentInstance.from_dict(json_item)


def load_segmentation_instances(
    datapath: str,
) -> Generator[SentenceSplitInstance, None, None]:
    # DO NOT CHANGE THIS
    with open(datapath, "r", encoding="utf8") as infile:
        json_list = json.load(infile)
        for json_item in json_list:
            yield SentenceSplitInstance.from_dict(json_item)


############################################################
# Class to represent classification instances
class ClassificationInstance:
    # DO NOT CHANGE THIS
    def __init__(self, label: str, features: List[str]) -> object:
        self.label = label
        self.features = features


############################################################
# Portion of the assignment that you should fill in

def accuracy(predictions: Sequence[str], expected: Sequence[str]) -> float:
    if len(predictions) == 0 or len(expected) == 0 or len(predictions)!= len(expected):
        raise ValueError("ValueError exception thrown: invalid input")
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == expected[i]:
            count += 1
    return count/len(predictions)

def recall(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    if len(predictions) == 0 or len(expected) == 0 or len(predictions) != len(expected):
        raise ValueError("ValueError exception thrown: invalid input")
    posLabels = 0
    true_positive = 0
    for i in range(len(expected)):
        if expected[i] == label:
            posLabels += 1  # the number of items should have the positive label
            if expected[i] == predictions[i]:
                true_positive += 1
    if posLabels == 0:
        return 0.0
    return true_positive/posLabels

def precision(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    if len(predictions) == 0 or len(expected) == 0 or len(predictions) != len(expected):
        raise ValueError("ValueError exception thrown: invalid input")
    true_positive = 0
    pred_positive = 0
    for i in range(len(predictions)):
        if predictions[i] == label:
            pred_positive += 1  # the number of items we predicted as positive
            if expected[i] == predictions[i]:
                true_positive += 1
    if pred_positive == 0:
        return 0.0
    return true_positive/pred_positive

def f1(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    if len(predictions) == 0 or len(expected) == 0 or len(predictions) != len(expected):
        raise ValueError("ValueError exception thrown: invalid input")
    prec = precision(predictions, expected, label)
    rec = recall(predictions, expected, label)
    if prec+rec == 0.0:
        return 0.0
    return 2*prec*rec/(prec+rec)

class BaselineAirlineSentimentFeatureExtractor:
    def extract_features(self, instance: AirlineSentimentInstance) -> ClassificationInstance:
        label = instance.label
        sentences = instance.sentences
        if len(sentences) == 0:
            raise ValueError("ValueError exception thrown: invalid input")
        dict = defaultdict()
        for s in sentences:
            for word in s:
                dict[word.lower()] = 1  # give a random value for the key
        l: List[str] = list(dict.keys())
        x = ClassificationInstance(label, l)
        return x

class BaselineSegmentationFeatureExtractor:
    def extract_features(self, instance: SentenceSplitInstance) -> ClassificationInstance:
        label = instance.label
        left_context = instance.left_context
        right_context = instance.right_context
        token = instance.token
        features = ["split_tok=" + token, "right_tok=" + right_context, "left_tok=" + left_context]
        x = ClassificationInstance(label, features)
        return x

class InstanceCounter:
    def __init__(self) -> None:
        self.size = 0  # number of instances
        self.feature_num = defaultdict(int)  # unique label ==> total count of all the features (not unique) seen with
        # the label
        self.label_counts = defaultdict(int)  # label ==> frequency
        self.feature_words = defaultdict(int)  # unique word ==> frequency
        self.feature_counts = defaultdict(lambda: defaultdict(int))  # label  ==> feature_word ==> frequency

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        for ins in instances:
            self.size += 1
            self.label_counts[ins.label] += 1
            self.feature_num[ins.label] += len(ins.features)
            for word in ins.features:
                self.feature_words[word] += 1
                self.feature_counts[ins.label][word] += 1

    def label_count(self, label: str) -> int:
        return self.label_counts[label]

    def total_labels(self) -> int:
        return self.size

    def conditional_feature_count(self, label: str, feature: str) -> int:
        return self.feature_counts[label][feature]

    def labels(self) -> List[str]:
        return list(self.label_counts.keys())

    def feature_vocab_size(self) -> int:
        return len(self.feature_words)

    def total_feature_count_for_class(self, label: str) -> int:
        return self.feature_num[label]


class NaiveBayesClassifier:
    def __init__(self, k):
        self.model = InstanceCounter()
        self.k = k

    def train(self, instances: Iterable[ClassificationInstance]) -> None:
        # Get probabilities through counting events and labels
        self.model.count_instances(instances)

    def classify(self, features: List[str]) -> str:
        label_dict = dict()
        for label in self.model.label_counts.keys():
            label_dict[label] = self.prob(features, label)
        label_list = label_dict.items()
        max_label = max(label_list, key=lambda x: x[1])
        return str(max_label[0])

    def prob(self, features: List[str], label: str) -> float:
        #  probability is in the log-space
        prob_sum = self.prior_prob(label)
        for word in features:
            prob_sum += self.likelihood_prob(word, label)
        return prob_sum

    def prior_prob(self, label: str) -> float:
        #  return the probability in the log-space
        if self.model.size == 0:
            raise ValueError("ValueError exception thrown: invalid input")
        return math.log(self.model.label_counts[label]/self.model.size)

    def likelihood_prob(self, feature: str, label) -> float:
        #  return the probability in the log-space
        feature_k = self.model.conditional_feature_count(label, feature)+self.k
        all_feature = self.model.total_feature_count_for_class(label)
        if all_feature == 0:
            return 0.0
        return math.log(feature_k/(all_feature+self.k*self.model.feature_vocab_size()))

    def test(self, instances: Iterable[ClassificationInstance]) -> Tuple[List[str], List[str]]:
        expected = list()
        predicted = list()
        for ins in instances:
            expected.append(ins.label)
            predicted.append(self.classify(ins.features))
        return predicted, expected

class TunedSegmentationFeatureExtractor:
    def __init__(self):
        self.k = None

    def extract_features(self, instance: SentenceSplitInstance) -> ClassificationInstance:
        pass
