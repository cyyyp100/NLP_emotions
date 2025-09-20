"""Training utilities for the bag-of-words neural network baseline.

This module exposes a small command line interface that can be used to
vectorise the ``data/text.csv`` dataset, train a dense neural network and
export the resulting artefacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

DEFAULT_CLASSES = (
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise",
)


def load_dataset(path: Path, sample_size: int | None = None) -> pd.DataFrame:
    """Load the tweet dataset.

    Parameters
    ----------
    path:
        Path to a CSV file containing at least two columns: ``text`` and
        ``label``.
    sample_size:
        Optional number of rows to keep from the top of the dataset. This is
        handy for quick experiments on limited hardware.
    """
    df = pd.read_csv(path)
    missing_columns = {"text", "label"} - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset at {path} is missing required columns: {missing_columns}"
        )
    if sample_size is not None:
        df = df.head(sample_size)
    return df.reset_index(drop=True)


def tokenize_texts(texts: Sequence[str]) -> List[List[str]]:
    """Tokenise raw strings into whitespace separated tokens."""

    return [str(text).split() for text in texts]


def vectorise_corpus(
    tokenised_texts: Sequence[Sequence[str]],
    *,
    vectorizer: CountVectorizer | None = None,
) -> Tuple[np.ndarray, CountVectorizer]:
    """Vectorise pre-tokenised texts.

    When ``vectorizer`` is provided the function reuses the fitted instance,
    otherwise a new ``CountVectorizer`` is created and fitted from scratch.
    """

    if vectorizer is None:
        vectorizer = CountVectorizer(
            lowercase=False,
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
        )
        features = vectorizer.fit_transform(tokenised_texts)
    else:
        features = vectorizer.transform(tokenised_texts)
    return features.toarray().astype("float32"), vectorizer


def build_model(input_dim: int, hidden_units: Iterable[int] = (128, 64), num_classes: int = 6) -> Sequential:
    """Create a simple fully connected neural network."""

    model = Sequential(name="bow_ann")
    first = True
    for units in hidden_units:
        if first:
            model.add(Dense(units, activation="relu", input_shape=(input_dim,)))
            first = False
        else:
            model.add(Dense(units, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train(
    data_path: Path,
    *,
    sample_size: int | None,
    test_size: float,
    batch_size: int,
    epochs: int,
    model_path: Path,
    vectorizer_path: Path,
    prediction_text: str | None = None,
) -> None:
    """Train the neural network and export the artefacts."""

    dataset = load_dataset(data_path, sample_size)
    tokenised = tokenize_texts(dataset["text"].astype(str))
    features, vectorizer = vectorise_corpus(tokenised)

    labels = dataset["label"].to_numpy(dtype=np.int32)
    num_classes = int(labels.max()) + 1

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    y_train_ohe = to_categorical(y_train, num_classes=num_classes)
    y_test_ohe = to_categorical(y_test, num_classes=num_classes)

    model = build_model(features.shape[1], num_classes=num_classes)
    model.fit(
        X_train,
        y_train_ohe,
        validation_data=(X_test, y_test_ohe),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test_ohe, verbose=0)
    print(f"Test accuracy: {accuracy:.3f} | loss: {loss:.3f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(vectorizer, vectorizer_path)

    if prediction_text:
        predicted_class, proba = predict_text(prediction_text, model, vectorizer)
        class_name = DEFAULT_CLASSES[predicted_class] if predicted_class < len(DEFAULT_CLASSES) else str(predicted_class)
        print(f"\nSample prediction for '{prediction_text}': {class_name} ({proba:.2%})")


def predict_text(
    text: str,
    model: Sequential,
    vectorizer: CountVectorizer,
) -> Tuple[int, float]:
    """Predict the emotion for a single piece of text."""

    tokenised = tokenize_texts([text])
    features, _ = vectorise_corpus(tokenised, vectorizer=vectorizer)
    probabilities = model.predict(features, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    return predicted_class, float(probabilities[predicted_class])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/text.csv"), help="Path to the CSV dataset.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Optional number of rows to keep for training. Use -1 to disable the limit.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data used for evaluation.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used during training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/modele_trained.keras"),
        help="Destination path for the trained Keras model.",
    )
    parser.add_argument(
        "--vectorizer-path",
        type=Path,
        default=Path("models/count_vectorizer.joblib"),
        help="Destination path for the fitted CountVectorizer.",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Optional text sample used to demonstrate inference once training is complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_size = None if args.sample_size is not None and args.sample_size < 0 else args.sample_size
    train(
        args.data,
        sample_size=sample_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        prediction_text=args.predict,
    )


if __name__ == "__main__":
    main()
