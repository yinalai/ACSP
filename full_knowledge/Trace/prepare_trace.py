from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


BREAKPOINTS = {
    3: (-0.43, 0.43),
    4: (-0.67, 0.0, 0.67),
    5: (-0.84, -0.25, 0.25, 0.84),
    6: (-0.97, -0.43, 0.0, 0.43, 0.97),
    7: (-1.07, -0.57, -0.18, 0.18, 0.57, 1.07),
    8: (-1.15, -0.67, -0.32, 0.0, 0.32, 0.67, 1.15),
    9: (-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22),
    10: (-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28),
}


def sax_compressed(series: np.ndarray, size: int = 4, window: int = 10) -> str:
    if size not in BREAKPOINTS or window <= 0:
        raise ValueError("SAX size must be 3..10 and window positive")
    raw = np.asarray(series, dtype=float).reshape(-1)
    if raw.size < window or not np.isfinite(raw).all():
        raise ValueError("Sequence too short or contains NaN/Inf")
    std = float(np.std(raw))
    if std == 0:
        raise ValueError("Constant series cannot be z-score transformed")
    z = (raw - float(np.mean(raw))) / std
    blocks = z[:(len(z) // window) * window].reshape(-1, window).mean(axis=1)
    
    
    syms = "abcdefghij"
    symbols = [syms[next((i for i, bound in enumerate(BREAKPOINTS[size])
                         if v <= bound), size - 1)] for v in blocks]
    return "".join(s for i, s in enumerate(symbols) if i == 0 or s != symbols[i - 1])


def convert(data_path: Path, label_path: Path, out_dir: Path,
            window: int = 10, test_size: int = 1000, split_seed: int = 2023) -> None:
    data = np.load(data_path, allow_pickle=False)
    labels = np.load(label_path, allow_pickle=False).reshape(-1).astype(int)
    if data.ndim != 2 or len(data) != len(labels):
        raise ValueError("Expected a 2D array of equal-length sequences and matched labels")
    if set(labels) != {1, 2, 3}:
        raise ValueError("Trace classes must be {1,2,3}; preprocess class 4 as in the original")
    if not 1 <= test_size < len(data):
        raise ValueError("test-size must be between 1 and N-1")
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=test_size, random_state=split_seed)
    outputs = {
        "data.txt": [sax_compressed(row, 4, window) for row in train_x],
        "label.txt": list(map(str, train_y)),
        "test_data.txt": [sax_compressed(row, 4, window) for row in test_x],
        "test_label.txt": list(map(str, test_y)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, entries in outputs.items():
        (out_dir / filename).write_text("\n".join(entries) + "\n", encoding="utf8")
    print(f"Saved {len(train_x)} train and {len(test_x)} test sequences to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Original Trace 4-symbol SAX file preparation")
    parser.add_argument("--data-npy", type=Path, required=True)
    parser.add_argument("--label-npy", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--paa-window", type=int, default=10)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=2023)
    args = parser.parse_args()
    convert(args.data_npy, args.label_npy, args.out_dir,
            args.paa_window, args.test_size, args.split_seed)


if __name__ == "__main__":
    main()
