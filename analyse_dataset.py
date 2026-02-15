"""
analyse_dataset.py
------------------
Small utility to inspect the synthetic sports/politics dataset and print
basic statistics that help understand class balance, vocabulary overlap,
and common/distinctive words for each label.

This file is intended for quick, human-readable dataset insights.
"""

import json
from collections import Counter
from feature_extraction import FeatureExtractor


def analyze_dataset():
    """Load `dataset.json` and print a series of summary statistics.

    Sections printed:
    - Class distribution
    - Vocabulary statistics (unique, shared, distinctive)
    - Document length statistics
    - Top words per class and some sample documents
    """
    # Load dataset from disk
    with open('dataset.json', 'r') as f:
        data = json.load(f)

    # Separate texts by label for class-specific analysis
    sports_texts = []
    politics_texts = []

    for item in data:
        if item['label'] == 'sports':
            sports_texts.append(item['text'])
        else:
            politics_texts.append(item['text'])

    # Header
    print("=" * 60)
    print("Dataset Analysis")
    print("=" * 60)

    # 1) Class distribution
    print("\n1. CLASS DISTRIBUTION")
    print(f"Total samples: {len(data)}")
    print(f"Sports samples: {len(sports_texts)}")
    print(f"Politics samples: {len(politics_texts)}")
    print(f"Balance: {len(sports_texts) / len(data) * 100:.1f}% sports, {len(politics_texts) / len(data) * 100:.1f}% politics")

    # Build vocabulary using the shared FeatureExtractor utilities
    extractor = FeatureExtractor()
    extractor.build_vocabulary(sports_texts + politics_texts)

    # 2) Vocabulary stats
    print("\n2. VOCABULARY")
    print(f"Total unique words: {len(extractor.vocabulary)}")

    sports_words = set()
    politics_words = set()

    for text in sports_texts:
        tokens = extractor.tokenize(text)
        sports_words.update(tokens)

    for text in politics_texts:
        tokens = extractor.tokenize(text)
        politics_words.update(tokens)

    sports_only = sports_words - politics_words
    politics_only = politics_words - sports_words
    common = sports_words.intersection(politics_words)

    print(f"Sports-only words: {len(sports_only)}")
    print(f"Politics-only words: {len(politics_only)}")
    print(f"Common words: {len(common)}")

    # 3) Document length statistics
    print("\n3. DOCUMENT LENGTH")
    sports_lengths = [len(extractor.tokenize(t)) for t in sports_texts]
    politics_lengths = [len(extractor.tokenize(t)) for t in politics_texts]

    print(f"Sports - Avg: {sum(sports_lengths) / len(sports_lengths):.1f} words, Min: {min(sports_lengths)}, Max: {max(sports_lengths)}")
    print(f"Politics - Avg: {sum(politics_lengths) / len(politics_lengths):.1f} words, Min: {min(politics_lengths)}, Max: {max(politics_lengths)}")

    # 4/5) Top words per class
    print("\n4. TOP SPORTS WORDS")
    sports_word_freq = Counter()
    for text in sports_texts:
        tokens = extractor.tokenize(text)
        sports_word_freq.update(tokens)

    for word, count in sports_word_freq.most_common(15):
        print(f"  {word}: {count}")

    print("\n5. TOP POLITICS WORDS")
    politics_word_freq = Counter()
    for text in politics_texts:
        tokens = extractor.tokenize(text)
        politics_word_freq.update(tokens)

    for word, count in politics_word_freq.most_common(15):
        print(f"  {word}: {count}")

    # 6/7) Distinctive words for each class (occurring at least 3 times)
    print("\n6. DISTINCTIVE SPORTS WORDS (sports only)")
    sports_distinctive = [w for w in sports_only if sports_word_freq[w] >= 3]
    print(f"  {', '.join(sorted(sports_distinctive)[:20])}")

    print("\n7. DISTINCTIVE POLITICS WORDS (politics only)")
    politics_distinctive = [w for w in politics_only if politics_word_freq[w] >= 3]
    print(f"  {', '.join(sorted(politics_distinctive)[:20])}")

    # 8) Print example documents for quick inspection
    print("\n8. SAMPLE DOCUMENTS")
    print("\nSports example:")
    print(f"  {sports_texts[0]}")
    print("\nPolitics example:")
    print(f"  {politics_texts[0]}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    analyze_dataset()