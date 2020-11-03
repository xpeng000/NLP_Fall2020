from abc import abstractmethod, ABC
from math import log
from operator import itemgetter
from typing import (
    Sequence,
    List,
    Tuple,
    Generator,
    Iterable,
    Any,
    Dict,
)

# hw4.py
# Version 1.0
# 10/21/2020

###############################################################################
# Course-provided classes, do not modify!


class Token:
    """A class that stores a token's text token text and POS tag together."""

    # DO NOT MODIFY THIS CLASS

    def __init__(self, token: str, tag: str):
        # DO NOT MODIFY THIS METHOD
        self.text = token
        self.tag = tag

    def __str__(self):
        # DO NOT MODIFY THIS METHOD
        return f"{self.text}/{self.tag}"

    def __repr__(self):
        # DO NOT MODIFY THIS METHOD
        return f"<Token {str(self)}>"

    def __eq__(self, other: Any):
        # DO NOT MODIFY THIS METHOD
        return (
            isinstance(other, Token)
            and self.text == other.text
            and self.tag == other.tag
        )

    def __lt__(self, other: "Token"):
        # DO NOT MODIFY THIS METHOD
        return self.to_tuple() < other.to_tuple()

    def __hash__(self):
        # DO NOT MODIFY THIS METHOD
        return hash(self.to_tuple())

    def to_tuple(self):
        """ Convert an instance of Token to a tuple of the token's text and its tag.
        Example:
            >>> token = Token("apple", "NN")
            >>> token.to_tuple()
            ("apple", "NN")
        """
        # DO NOT MODIFY THIS METHOD
        return self.text, self.tag

    @staticmethod
    def from_tuple(t: Tuple[str, ...]):
        """Create a Token object from a tuple. """
        # DO NOT MODIFY THIS METHOD
        assert len(t) == 2
        return Token(t[0], t[1])

    @staticmethod
    def from_string(s: str) -> "Token":
        """Create a Token object from a string with the format 'token/tag'.

        Usage: Token.from_string("cat/NN")
        """
        # DO NOT MODIFY THIS METHOD
        return Token(*s.rsplit("/", 1))


class Tagger(ABC):
    # DO NOT MODIFY THIS CLASS

    @abstractmethod
    def train(self, sentences: Iterable[Sequence[Token]]) -> None:
        """Train the part of speech tagger by collecting needed counts from sentences."""
        # DO NOT IMPLEMENT THIS METHOD HERE
        # Instead, implement it in the subclasses.
        raise NotImplementedError

    @abstractmethod
    def tag_sentence(self, sentence: Sequence[str]) -> List[str]:
        """Tag a sentence with part of speech tags.

        Sample usage:
             tag_sentence(["I", "ate", "an", "apple"])
             returns: ["PRP", "VBD", "DT", "NN"]
        """
        # DO NOT IMPLEMENT THIS METHOD HERE
        # Instead, implement it in the subclasses.
        raise NotImplementedError

    def tag_sentences(
        self, sentences: Iterable[Sequence[str]]
    ) -> Generator[List[str], None, None]:
        """Yield a list of tags for each sentence."""
        # DO NOT MODIFY THIS METHOD
        for sentence in sentences:
            yield self.tag_sentence(sentence)

    def test(
        self, tagged_sents: Iterable[Sequence[Token]]
    ) -> Tuple[List[str], List[str]]:
        """Run the tagger and return a tuple of predicted and expected tag lists.

        The predicted and actual tags can then be used for calculating accuracy or other
        metrics. This does not preserve sentence boundaries.
        """
        # DO NOT MODIFY THIS METHOD
        predicted: List[str] = []
        actual: List[str] = []
        for sent in tagged_sents:
            predicted.extend(self.tag_sentence([t.text for t in sent]))
            actual.extend([t.tag for t in sent])
        return predicted, actual


# Use these functions in your Bigram Taggers
def _safe_log(n: float) -> float:
    """Return the natural log of n or -inf if n is 0.0."""
    # DO NOT MODIFY THIS METHOD
    return float("-inf") if n == 0.0 else log(n)


def _max_item(scores: Dict[str, float]) -> Tuple[str, float]:
    """Given a dict of tag: score, return a tuple of the max tag and its score."""
    # PyCharm may give a false positive type warning, ignore it
    return max(scores.items(), key=itemgetter(1))


###############################################################################
# Solution begins here, modify the classes below


class MostFrequentTagTagger(Tagger):
    def __init__(self):
        # DO NOT MODIFY THIS METHOD
        self.default_tag = None

    def train(self, sentences: Iterable[Sequence[Token]]) -> None:
        pass

    def tag_sentence(self, sentence: Sequence[str]) -> List[str]:
        pass


class UnigramTagger(Tagger):
    def __init__(self):
        # Add Counters/dicts/defaultdicts/etc. that you need here.
        pass

    def train(self, sentences: Iterable[Sequence[Token]]):
        pass

    def tag_sentence(self, sentence: Sequence[str]) -> List[str]:
        pass


class SentenceCounter:
    def __init__(self, k):
        self.k = k
        # Add Counters/dicts/defaultdicts/etc. that you need here.
        pass

    def count_sentences(self, sentences: Iterable[Sequence[Token]]) -> None:
        """ Count token text and tags in sentences.

        After this function runs the SentenceCounter object should be ready
        to return values for initial, transition, and emission probabilities
        as well as return the sorted tagset.
        """
        pass

    def tagset(self) -> List[str]:
        """Return a sorted list of the unique tags."""
        pass

    def emission_prob(self, tag: str, word: str) -> float:
        pass

    def transition_prob(self, prev_tag: str, current_tag: str) -> float:
        pass

    def initial_prob(self, tag: str) -> float:
        pass


class BigramTagger(Tagger, ABC):
    # You can add additional methods to this class if you want to share anything
    # between the greedy and Viterbi taggers. However, do not modify any of the
    # implemented methods.
    def __init__(self, k) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter = SentenceCounter(k)

    def train(self, sents: Iterable[Sequence[Token]]) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter.count_sentences(sents)

    def sequence_probability(
        self, sentence: Sequence[str], tags: Sequence[str]
    ) -> float:
        """
        Compute the probability for the sequence of tags
        with the sequence of tokens in the sentence.
        """
        pass


class GreedyBigramTagger(BigramTagger):
    # DO NOT DEFINE __init__ or train

    def tag_sentence(self, sentence: Sequence[str]) -> List[str]:
        pass


class ViterbiBigramTagger(BigramTagger):
    # DO NOT DEFINE __init__ or train

    def tag_sentence(self, sentence: Sequence[str]) -> List[str]:
        pass
