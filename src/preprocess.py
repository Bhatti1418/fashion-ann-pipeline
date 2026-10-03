"""Normalize Fashion-MNIST arrays and create a stratified validation split."""

from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Load raw arrays, apply configured preprocessing, and save the splits."""
    with (ROOT / "params.yaml").open(encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)
    preprocess_params = params["preprocess"]

    raw_dir = ROOT / "data" / "raw"
    train_data = np.load(raw_dir / "train.npz")
    test_data = np.load(raw_dir / "test.npz")
    x_train = train_data["images"].astype(np.float32) / 255.0
    y_train = train_data["labels"].astype(np.int64)
    x_test = test_data["images"].astype(np.float32) / 255.0
    y_test = test_data["labels"].astype(np.int64)

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=preprocess_params["test_size"],
        random_state=preprocess_params["seed"],
        stratify=y_train,
    )

    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "train.npz", images=x_fit, labels=y_fit)
    np.savez_compressed(output_dir / "val.npz", images=x_val, labels=y_val)
    np.savez_compressed(output_dir / "test.npz", images=x_test, labels=y_test)
    print(
        f"Saved {len(y_fit)} training, {len(y_val)} validation, "
        f"and {len(y_test)} test examples to {output_dir}"
    )


if __name__ == "__main__":
    main()