"""
demo.py
-------
Simple interactive demo that trains a quick MultinomialNB classifier on the
top portion of `dataset.json` and allows a user to type sentences to see the
predicted label and class probabilities.
"""

from feature_extraction import FeatureExtractor, load_dataset, split_data
from sklearn.naive_bayes import MultinomialNB
import json


def train_demo_classifier():
    """Train a lightweight classifier on the training split and return the
    classifier along with the extractor used for feature vectorization.
    """
    texts, labels = load_dataset('dataset.json')
    train_texts, train_labels, _, _ = split_data(texts, labels)

    extractor = FeatureExtractor()
    extractor.build_vocabulary(train_texts)
    train_features = extractor.bag_of_words(train_texts)

    classifier = MultinomialNB()
    classifier.fit(train_features, train_labels)

    return classifier, extractor


def predict_text(text, classifier, extractor):
    """Return predicted label and probability vector for a single input text."""
    features = extractor.bag_of_words([text])
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]

    return prediction, probabilities


def main():
    print("Loading model...")
    classifier, extractor = train_demo_classifier()
    print("Model loaded successfully!\n")

    print("=" * 60)
    print("Sports vs Politics Classifier Demo")
    print("=" * 60)
    print("\nEnter text to classify (or 'quit' to exit)")
    print("\nExample inputs:")
    print("- The team won the championship")
    print("- Parliament passed new legislation")
    print()

    while True:
        user_input = input("Enter text: ").strip()

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if len(user_input) == 0:
            print("Please enter some text.\n")
            continue

        prediction, probabilities = predict_text(user_input, classifier, extractor)

        print("\nPrediction:", prediction.upper())
        print("Confidence:")
        # Note: ordering of probabilities follows label ordering used during training
        print("  Politics: {:.2%}".format(probabilities[0]))
        print("  Sports:   {:.2%}".format(probabilities[1]))
        print()


if __name__ == '__main__':
    main()