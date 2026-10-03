"""Download Fashion-MNIST and persist the unmodified arrays."""

from pathlib import Path

import numpy as np
from tensorflow.keras.datasets import fashion_mnist


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Download Fashion-MNIST and save train/test arrays under data/raw."""
    output_dir = ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    np.savez_compressed(
        output_dir / "train.npz", images=x_train, labels=y_train
    )
    np.savez_compressed(
        output_dir / "test.npz", images=x_test, labels=y_test
    )
    print(f"Saved {len(x_train)} training and {len(x_test)} test images to {output_dir}")


if __name__ == "__main__":
    main()