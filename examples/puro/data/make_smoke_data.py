#!/usr/bin/env python3
"""Generate a tiny synthetic PROM packed-NPY dataset for environment validation.

Same on-disk contract as build_puro_data.py, so a smoke run exercises the real
CustomGPTDataset / THD packing path without waiting on the corpus download.
Loss values from this data are meaningless by construction.
"""
import argparse
import json
from pathlib import Path

import numpy as np

EOS_ID = 151645
VOCAB_SIZE = 151936


def build(out_dir: Path, rows: int, seq_len: int, shards: int, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    per = max(1, rows // shards)
    file_list, file_len = [], []
    for i in range(shards):
        n = per if i < shards - 1 else rows - per * (shards - 1)
        if n <= 0:
            continue
        arr = rng.integers(0, VOCAB_SIZE, size=(n, seq_len + 1), dtype=np.int32)
        # Scatter EOS so the THD path sees realistic variable-length documents.
        for r in range(n):
            cuts = rng.integers(64, 1024, size=8).cumsum()
            arr[r, cuts[cuts < seq_len]] = EOS_ID
        name = f"shard_{i:05d}.npy"
        np.save(out_dir / name, arr)
        file_list.append(name)
        file_len.append(int(n))
    meta = {
        "size": sum(file_len),
        "file_list": file_list,
        "file_len": file_len,
        "sample_index_offset": 0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{out_dir}: {meta['size']} samples x {seq_len+1} tokens in {len(file_list)} shards")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="parent dir; train/ and valid/ created under it")
    ap.add_argument("--train-rows", type=int, default=8192)
    ap.add_argument("--valid-rows", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--shards", type=int, default=4)
    a = ap.parse_args()
    build(Path(a.out_dir) / "train", a.train_rows, a.seq_len, a.shards, 0)
    build(Path(a.out_dir) / "valid", a.valid_rows, a.seq_len, 1, 1)
