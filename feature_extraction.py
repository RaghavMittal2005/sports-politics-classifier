"""
feature_extraction.py
---------------------
Provides a lightweight `FeatureExtractor` class with simple tokenization,
bag-of-words, TF-IDF and n-gram extraction utilities used by the demos and
experiments in this repository.
"""

import math
import json
from collections import Counter


class FeatureExtractor:
    """Simple feature extraction helper for small text datasets.

    Attributes
    - `vocabulary`: set of tokens collected from training documents
    - `idf_values`: computed IDF values for words (used by `tfidf`)
    - `ngram_vocabulary`: dict mapping n -> sorted list of n-grams
    """

    def __init__(self):
        self.vocabulary = set()
        self.idf_values = {}
        self.ngram_vocabulary = {}

    def tokenize(self, text):
        """Lowercase and split text into alphanumeric tokens.

        This is intentionally lightweight and intended for examples rather
        than production NLP preprocessing.
        """
        text = text.lower()
        tokens = text.split()
        cleaned = []
        for token in tokens:
            word = ""
            for char in token:
                if char.isalnum():
                    word += char
            if len(word) > 0:
                cleaned.append(word)
        return cleaned

    def build_vocabulary(self, documents):
        """Populate `self.vocabulary` from an iterable of documents."""
        for doc in documents:
            tokens = self.tokenize(doc)
            for token in tokens:
                self.vocabulary.add(token)
        return list(self.vocabulary)

    def bag_of_words(self, documents):
        """Return integer count vectors for each document using the current vocabulary."""
        vocab = sorted(list(self.vocabulary))
        features = []

        for doc in documents:
            tokens = self.tokenize(doc)
            word_count = {}
            for token in tokens:
                if token not in word_count:
                    word_count[token] = 0
                word_count[token] += 1

            feature_vector = []
            for word in vocab:
                if word in word_count:
                    feature_vector.append(word_count[word])
                else:
                    feature_vector.append(0)
            features.append(feature_vector)

        return features

    def compute_idf(self, documents):
        """Compute IDF for each word in vocabulary based on provided documents."""
        vocab = list(self.vocabulary)
        doc_count = len(documents)

        for word in vocab:
            docs_with_word = 0
            for doc in documents:
                tokens = self.tokenize(doc)
                if word in tokens:
                    docs_with_word += 1

            if docs_with_word > 0:
                self.idf_values[word] = math.log(doc_count / docs_with_word)
            else:
                self.idf_values[word] = 0

    def tfidf(self, documents):
        """Return TF-IDF feature vectors for `documents` using computed IDF."""
        vocab = sorted(list(self.vocabulary))
        features = []

        self.compute_idf(documents)

        for doc in documents:
            tokens = self.tokenize(doc)
            token_count = len(tokens)

            word_freq = {}
            for token in tokens:
                if token not in word_freq:
                    word_freq[token] = 0
                word_freq[token] += 1

            feature_vector = []
            for word in vocab:
                tf = 0
                if word in word_freq:
                    tf = word_freq[word] / token_count

                idf = self.idf_values.get(word, 0)
                tfidf_value = tf * idf
                feature_vector.append(tfidf_value)

            features.append(feature_vector)

        return features

    def extract_ngrams(self, text, n):
        """Return list of n-gram strings extracted from a single text."""
        tokens = self.tokenize(text)
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i + n])
            ngrams.append(ngram)
        return ngrams

    def build_ngram_vocabulary(self, documents, n):
        """Build and cache the n-gram vocabulary for `n` across `documents`."""
        if n not in self.ngram_vocabulary:
            ngram_vocab = set()
            for doc in documents:
                ngrams = self.extract_ngrams(doc, n)
                for ng in ngrams:
                    ngram_vocab.add(ng)
            self.ngram_vocabulary[n] = sorted(list(ngram_vocab))
        return self.ngram_vocabulary[n]

    def ngram_features(self, documents, n):
        """Return count vectors over the cached n-gram vocabulary for each document."""
        ngram_vocab_sorted = self.ngram_vocabulary.get(n, [])
        features = []

        for doc in documents:
            ngrams = self.extract_ngrams(doc, n)
            ngram_count = {}
            for ng in ngrams:
                if ng not in ngram_count:
                    ngram_count[ng] = 0
                ngram_count[ng] += 1

            feature_vector = []
            for ng in ngram_vocab_sorted:
                if ng in ngram_count:
                    feature_vector.append(ngram_count[ng])
                else:
                    feature_vector.append(0)

            features.append(feature_vector)

        return features


def load_dataset(filepath):
    """Load a dataset saved as a list of `{'text','label'}` dicts."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    texts = []
    labels = []
    for item in data:
        texts.append(item['text'])
        labels.append(item['label'])

    return texts, labels


def split_data(texts, labels, train_ratio=0.8):
    """Simple split: first `train_ratio` portion for training, rest for testing.

    Note: this preserves order (no shuffle) which is fine for examples but
    not recommended for robust experimental pipelines.
    """
    total = len(texts)
    train_size = int(total * train_ratio)

    train_texts = texts[:train_size]
    train_labels = labels[:train_size]
    test_texts = texts[train_size:]
    test_labels = labels[train_size:]

    return train_texts, train_labels, test_texts, test_labels