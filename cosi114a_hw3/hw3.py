# test_hw3.py
# Version 1.0
# 10/6/2020

import json
from collections import defaultdict
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
    def __init__(self, label: str, features: List[str]):
        self.label = label
        self.features = features


############################################################
# Portion of the assignment that you should fill in


def accuracy(predictions: Sequence[str], expected: Sequence[str]) -> float:
    pass


def recall(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    pass


def precision(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    pass


def f1(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    pass


class BaselineAirlineSentimentFeatureExtractor:
    def extract_features(
        self, instance: AirlineSentimentInstance
    ) -> ClassificationInstance:
        pass


class BaselineSegmentationFeatureExtractor:
    def extract_features(
        self, instance: SentenceSplitInstance
    ) -> ClassificationInstance:
        pass


class InstanceCounter:
    def __init__(self) -> None:
        pass

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        pass

    def label_count(self, label: str) -> int:
        pass

    def total_labels(self) -> int:
        pass

    def conditional_feature_count(self, label: str, feature: str) -> int:
        pass

    def labels(self) -> List[str]:
        pass

    def feature_vocab_size(self) -> int:
        pass

    def total_feature_count_for_class(self, label: str) -> int:
        pass


class NaiveBayesClassifier:
    def __init__(self, k):
        pass

    def train(self, instances: Iterable[ClassificationInstance]) -> None:
        pass

    def classify(self, features: List[str]) -> str:
        pass

    def prob(self, features: List[str], label: str) -> float:
        pass

    def prior_prob(self, label: str) -> float:
        pass

    def likelihood_prob(self, feature: str, label) -> float:
        pass

    def test(
        self, instances: Iterable[ClassificationInstance]
    ) -> Tuple[List[str], List[str]]:
        pass


class TunedSegmentationFeatureExtractor:
    def __init__(self):
        self.k = None

    def extract_features(
        self, instance: SentenceSplitInstance
    ) -> ClassificationInstance:
        pass
