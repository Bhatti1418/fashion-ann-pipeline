"""Train a fully connected ANN and save its model and epoch history."""

import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Build, train, and persist the configured Fashion-MNIST ANN."""
    with (ROOT / "params.yaml").open(encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)
    train_params = params["train"]
    tf.keras.utils.set_random_seed(train_params["seed"])

    train_data = np.load(ROOT / "data" / "processed" / "train.npz")
    val_data = np.load(ROOT / "data" / "processed" / "val.npz")
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28)),
            keras.layers.Flatten(),
            keras.layers.Dense(train_params["dense_units"], activation="relu"),
            keras.layers.Dropout(train_params["dropout_rate"]),
            keras.layers.Dense(10, activation="softmax"),
        ],
        name="fashion_mnist_ann",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=train_params["learning_rate"]
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_data["images"],
        train_data["labels"],
        validation_data=(val_data["images"], val_data["labels"]),
        epochs=train_params["epochs"],
        batch_size=train_params["batch_size"],
        verbose=2,
    )

    output_dir = ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.h5")
    history_columns = list(history.history)
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as history_file:
        writer = csv.writer(history_file)
        writer.writerow(history_columns)
        writer.writerows(
            zip(*(history.history[column] for column in history_columns))
        )
    print(f"Saved trained model and history to {output_dir}")


if __name__ == "__main__":
    main()