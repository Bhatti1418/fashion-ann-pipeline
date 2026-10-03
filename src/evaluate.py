"""Evaluate the trained ANN and save metrics and a confusion matrix."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix
from tensorflow import keras


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def main() -> None:
    """Compute test metrics and persist the confusion-matrix visualization."""
    with (ROOT / "params.yaml").open(encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)
    test_data = np.load(ROOT / "data" / "processed" / "test.npz")
    model = keras.models.load_model(ROOT / "models" / "model.h5")
    test_loss, test_accuracy = model.evaluate(
        test_data["images"], test_data["labels"], verbose=0
    )
    predictions = np.argmax(
        model.predict(test_data["images"], verbose=0), axis=1
    )
    matrix = confusion_matrix(
        test_data["labels"], predictions, labels=np.arange(len(CLASS_NAMES))
    )

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_samples": int(len(test_data["labels"])),
        "class_names": CLASS_NAMES,
        "confusion_matrix": matrix.tolist(),
        "train_params": params["train"],
    }
    (ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    output_path = ROOT / "reports" / "confusion_matrix.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Fashion-MNIST Test Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.4f}")
    print(f"Saved metrics.json and {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()