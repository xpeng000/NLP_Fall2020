#! /usr/bin/env python

"""
This is a script for debugging your homework functions.
You can also use it to explore the data provided and look around at n-gram probabilities
for different words for different genres
"""

import os
from collections import Counter

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


def main():
    # Recommend commenting out
    debug_functions()
    explore_mini_corpus()


def debug_functions():
    # Prints out small output for each function in hw1
    # You can modify this to debug your functions

    # Generate sentences by loading from test data
    # Store in a list since will use multiple times
    news_sents = list(gen_sentences(os.path.join("test_data", "brown-news.txt")))

    # Generate unigrams, bigrams, trigrams --------------------------------------------------------
    print("----------N-gram Generators----------")
    unigrams = list(gen_unigrams(news_sents))
    bigrams = list(gen_bigrams(news_sents))
    trigrams = list(gen_trigrams(news_sents))
    print(f"Unigrams:\n {unigrams[:8]}\n")
    print(f"Bigrams:\n {bigrams[:8]}\n")
    print(f"Trigrams:\n {trigrams[:8]}\n")
    print()

    # Counts --------------------------------------------------------------------------------------
    print("----------Counts----------")
    print("--Unigram Counts--")
    unigram_counts = count_unigrams(news_sents)
    for gram in list(unigram_counts)[:3]:
        print(f"{gram}: {unigram_counts[gram]}")
    print()
    print("--Bigram Counts--")
    bigram_counts = count_bigrams(news_sents)
    for gram in list(bigram_counts)[:3]:
        print(f"{gram}: {bigram_counts[gram]}")
    print()
    print("--Trigram Counts--")
    trigram_counts = count_trigrams(news_sents)
    for gram in list(trigram_counts)[:3]:
        print(f"{gram}: {trigram_counts[gram]}")
    print()

    # Frequency Distributions ---------------------------------------------------------------------
    print("----------Frequency Distributions ----------")
    print("Bigram Frequency Distribution")
    bigram_freq = bigram_freq_dist(news_sents)
    for word1 in list(bigram_freq)[:2]:
        print(f"\t{word1}: ")
        for word, count in Counter(bigram_freq[word1]).most_common(3):
            print(f"\t\t{word}: {count}")
    print()
    print("Trigram Frequency Distribution")
    trigram_freq = trigram_freq_dist(news_sents)
    for bigram in list(trigram_freq)[:2]:
        print(f"\t{bigram}: ")
        for word, count in Counter(trigram_freq[bigram]).most_common(3):
            print(f"\t\t{word}: {count}")
    print()

    # Probabilities -------------------------------------------------------------------------------
    print("----------Probabilities----------")
    print("Unigram probabilities")
    unigram_probs = unigram_probabilities(news_sents)
    for word in list(unigram_probs)[:3]:
        print(f"\t{word}: {unigram_probs[word]:.3f}")
    print()

    print("Bigram probabilities")
    bigram_probs = bigram_probabilities(news_sents)
    for word1 in list(bigram_probs)[:3]:
        print(f"\t{word1}:")
        for word2, prob in Counter(bigram_probs[word1]).most_common(3):
            print(f"\t\t{word2}: {prob:.5f}")
    print()

    print("Trigram probabilities")
    trigram_probs = trigram_probabilities(news_sents)
    for bigram in list(trigram_probs)[:3]:
        print(f"\t{bigram}:")
        for word, prob in Counter(trigram_probs[bigram]).most_common(3):
            print(f"\t\t{word}: {prob:.3f}")
    print()


def explore_mini_corpus():
    # Use this space to explore and compare the documents in the mini-corpus provided

    # Brown samples by genre
    news_sents = list(gen_sentences(os.path.join("test_data", "brown-news.txt")))
    humor_sents = list(gen_sentences(os.path.join("test_data", "brown-humor.txt")))
    sci_fi_sents = list(
        gen_sentences(os.path.join("test_data", "brown-science_fiction.txt"))
    )
    romance_sents = list(gen_sentences(os.path.join("test_data", "brown-romance.txt")))

    # Tweets
    ariana_sents = list(gen_sentences(os.path.join("test_data", "ArianaGrande.txt")))
    cristiano_sents = list(gen_sentences(os.path.join("test_data", "Cristiano.txt")))
    kingjames_sents = list(gen_sentences(os.path.join("test_data", "KingJames.txt")))
    gaga_sents = list(gen_sentences(os.path.join("test_data", "ladygaga.txt")))
    lizzo_sents = list(gen_sentences(os.path.join("test_data", "lizzo.txt")))

    # Examples:
    print("Lady Gaga")
    for word, count in Counter(unigram_probabilities(gaga_sents)).most_common(10):
        print(f"{word}: {count:.3f}")
    print()
    print("\nLebron James")
    for word, count in Counter(unigram_probabilities(kingjames_sents)).most_common(10):
        print(f"{word}: {count:.3f}")
    print()

    print("Difference between Lebron and Gaga unigram probabilities")
    diff = unigram_probabilities_difference(
        unigram_probabilities(gaga_sents, lower=True),
        unigram_probabilities(kingjames_sents, lower=True),
        intersection_only=False,
    )
    for word, val in Counter(diff).most_common(20):
        print(f"{word}: {val}")
    print()
    print('Difference between probabilities of word following "I" for Gaga and Lizzo')
    bigram_diff = bigram_probabilities_difference(
        bigram_probabilities(lizzo_sents, lower=True),
        bigram_probabilities(gaga_sents, lower=True),
        "i",
        intersection_only=True,
    )
    for word, val in Counter(bigram_diff).most_common(20):
        print(f"{word}: {val}")

    # TODO call your methods on some other collections of sentences
    #   What's the difference in unigram probabilities
    #   between ArianaGrande and sci-fi, for example?


if __name__ == "__main__":
    main()
