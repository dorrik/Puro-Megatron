#!/usr/bin/env python3
"""Build a PROM packed-NPY dataset for Puro-Megatron from the released Puro-2B corpus.

The training code (`CustomGPTDataset` in pretrain_gpt.py) expects a directory of

    metadata.json   {"size": N, "file_list": [...], "file_len": [...]}
    shard_00000.npy  int32 array of shape (rows, seq_len + 1)

where every row is a contiguous slice of a token stream in which documents are
separated by the EOS id. `--attention-type thd` reconstructs packed-sequence
metadata from those EOS positions at run time, so plain concatenation is the
correct layout.

Source corpus: https://huggingface.co/datasets/thu-pacman/Puro-2B (`phase1/`),
tokenized with the bundled `qwen2_tokenizer/` for exact token accounting.
Domain shares default to the published Phase 1 mixture.

Usage:
    python build_puro_data.py --out-dir $SCRATCH/puro/data --target-tokens 12.5e9
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "thu-pacman/Puro-2B"
TOKENIZER_SUBDIR = "qwen2_tokenizer"
EOS_ID = 151645  # <|im_end|>, matches --eos-token-id in the Puro recipes
VOCAB_SIZE = 151936

# Published Phase 1 domain mixture (dataset card "Dataset Summary" table).
PHASE1_SHARES = {"English": 0.732, "Chinese": 0.117, "Code": 0.079, "Math": 0.072}

# Directory -> domain. Only used to order downloads; the authoritative label is
# the per-row `domain_category` column, which is what the budgets are applied to.
COMPONENT_DOMAIN = {
    "fineweb-edu-dedup": "English",
    "cosmopedia-v2": "English",
    "arxiv": "English",
    "nemotron-high-quality": "English",
    "nemotron-high-quality-synthetic": "English",
    "fineweb-edu-chinese-dedup": "Chinese",
    "fineweb-edu-chinese-v2.2-3_4-top20": "Chinese",
    "chinesewebtext2-highquality-top50": "Chinese",
    "baidu-baike": "Chinese",
    "finewiki-zh": "Chinese",
    "alpaca-zh": "Chinese",
    "undl-zh2en-aligned": "Chinese",
    "swallow-code-v2": "Code",
    "python-edu": "Code",
    "nemotron-synthetic-code": "Code",
    "finemath": "Math",
    "swallow-math-v2": "Math",
    "nemotron-cc-math-v1-4plus": "Math",
}

_TOKENIZER = None


def get_tokenizer(tokenizer_dir: str):
    """Load the fast tokenizer once per worker process."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from tokenizers import Tokenizer

        _TOKENIZER = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
    return _TOKENIZER


def normalize_domain(raw: str) -> str:
    """Map the corpus' domain_category values onto the four budget buckets."""
    s = (raw or "").strip().lower()
    if s.startswith("en") or "english" in s:
        return "English"
    if s.startswith("zh") or "chinese" in s:
        return "Chinese"
    if "code" in s:
        return "Code"
    if "math" in s:
        return "Math"
    if "instruct" in s or "sft" in s:
        return "Instruct"
    return "Other"


# --------------------------------------------------------------------------
# Stage 1: choose and download source files
# --------------------------------------------------------------------------
def plan_downloads(target_tokens: float, cache_dir: str, phase: str) -> list[tuple[str, str]]:
    """Round-robin components within each domain until its token budget is met.

    Returns a list of (repo_path, domain) still in download order.
    """
    files = [f for f in list_repo_files(REPO_ID, repo_type="dataset") if f.startswith(f"{phase}/")]
    by_component: dict[str, list[str]] = defaultdict(list)
    for f in files:
        if f.endswith(".parquet"):
            by_component[f.split("/")[1]].append(f)
    for v in by_component.values():
        v.sort()

    plan: list[tuple[str, str]] = []
    for domain in PHASE1_SHARES:
        comps = sorted(c for c, d in COMPONENT_DOMAIN.items() if d == domain and by_component.get(c))
        if not comps:
            print(f"  !! no components for domain {domain}", file=sys.stderr)
            continue
        # Emit the full round-robin ordering. download() consumes it lazily and
        # stops on measured token counts, so no per-file size estimate is needed.
        idx = 0
        while any(idx < len(by_component[c]) for c in comps):
            for c in comps:
                if idx < len(by_component[c]):
                    plan.append((by_component[c][idx], domain))
            idx += 1
    return plan


def download(plan: list[tuple[str, str]], cache_dir: str, target_tokens: float) -> list[tuple[str, str]]:
    """Download planned files, stopping per-domain once real token counts suffice."""
    os.makedirs(cache_dir, exist_ok=True)
    got: dict[str, float] = defaultdict(float)
    kept: list[tuple[str, str]] = []
    for repo_path, domain in plan:
        if got[domain] >= target_tokens * PHASE1_SHARES[domain]:
            continue
        t0 = time.time()
        local = hf_hub_download(
            REPO_ID, repo_path, repo_type="dataset", local_dir=cache_dir
        )
        # token_count is a stored column; reading just it is cheap and exact.
        n = int(pq.read_table(local, columns=["token_count"])["token_count"].to_numpy().sum())
        got[domain] += n
        kept.append((local, domain))
        print(
            f"  {repo_path:60s} {n/1e9:6.3f}B  {domain:8s} "
            f"total={got[domain]/1e9:6.2f}B/{target_tokens*PHASE1_SHARES[domain]/1e9:.2f}B "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )
    return kept


# --------------------------------------------------------------------------
# Stage 2: tokenize into packed shards
# --------------------------------------------------------------------------
def tokenize_file(args) -> tuple[str, int, int, str]:
    """Tokenize one parquet file into one shard. Returns (shard, rows, tokens, domain)."""
    local_path, domain, out_dir, tokenizer_dir, seq_plus1, shard_name, token_budget = args
    # `tokenizers` parallelises encode_batch with rayon. With one process per
    # file that would oversubscribe the node by ~190x, so keep each worker
    # single-threaded and get the parallelism from the process pool.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    tok = get_tokenizer(tokenizer_dir)

    buf = np.empty(0, dtype=np.int32)
    rows: list[np.ndarray] = []
    produced = 0
    pf = pq.ParquetFile(local_path)
    stop = False
    for batch in pf.iter_batches(batch_size=2000, columns=["text", "domain_category"]):
        texts = batch.column("text").to_pylist()
        encs = tok.encode_batch_fast(texts) if hasattr(tok, "encode_batch_fast") else tok.encode_batch(texts)
        chunk: list[np.ndarray] = []
        for e in encs:
            ids = np.asarray(e.ids, dtype=np.int32)
            chunk.append(ids)
            chunk.append(np.array([EOS_ID], dtype=np.int32))
        if chunk:
            buf = np.concatenate([buf, *chunk])
        n_full = len(buf) // seq_plus1
        if n_full:
            take = buf[: n_full * seq_plus1].reshape(n_full, seq_plus1)
            rows.append(take)
            produced += n_full * seq_plus1
            buf = buf[n_full * seq_plus1 :]
        if token_budget and produced >= token_budget:
            stop = True
            break
    if not rows:
        return ("", 0, 0, domain)

    arr = np.concatenate(rows, axis=0)
    if token_budget:
        max_rows = max(1, int(token_budget // seq_plus1))
        arr = arr[:max_rows]
    assert arr.max() < VOCAB_SIZE, f"token id {arr.max()} >= vocab {VOCAB_SIZE}"
    out = Path(out_dir) / shard_name
    np.save(out, arr)
    return (shard_name, int(arr.shape[0]), int(arr.size), domain)


def write_metadata(out_dir: Path, shards: list[tuple[str, int]]):
    """shards: list of (filename, n_rows) in read order."""
    file_list = [s for s, _ in shards]
    file_len = [n for _, n in shards]
    meta = {
        "size": sum(file_len),
        "file_list": file_list,
        "file_len": file_len,
        "sample_index_offset": 0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="parent dir; train/ and valid/ are created under it")
    ap.add_argument("--cache-dir", default=None, help="where parquet downloads land (default: <out-dir>/_parquet)")
    ap.add_argument("--target-tokens", type=float, default=12.5e9)
    ap.add_argument("--valid-tokens", type=float, default=40e6)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--phase", default="phase1")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    cache = Path(args.cache_dir or out / "_parquet")
    train_dir, valid_dir = out / "train", out / "valid"
    for d in (train_dir, valid_dir, cache):
        d.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = args.tokenizer_dir
    if tokenizer_dir is None:
        tokenizer_dir = str(out / "tokenizer")
        Path(tokenizer_dir).mkdir(exist_ok=True)
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            p = hf_hub_download(REPO_ID, f"{TOKENIZER_SUBDIR}/{f}", repo_type="dataset", local_dir=str(out))
            shutil.copy(p, Path(tokenizer_dir) / f)
    print(f"tokenizer: {tokenizer_dir}")

    total = args.target_tokens + args.valid_tokens
    if args.skip_download:
        kept = []
        for p in sorted(cache.rglob("*.parquet")):
            kept.append((str(p), COMPONENT_DOMAIN.get(p.parent.name, "English")))
        print(f"reusing {len(kept)} cached parquet files")
    else:
        print(f"=== planning downloads for {total/1e9:.2f}B tokens ===")
        plan = plan_downloads(total, str(cache), args.phase)
        print(f"planned {len(plan)} files; downloading")
        kept = download(plan, str(cache), total)

    if not kept:
        sys.exit("no source files")

    # Per-domain token budgets, applied per file so the mixture is preserved.
    per_domain_files: dict[str, list[str]] = defaultdict(list)
    for path, domain in kept:
        per_domain_files[domain].append(path)

    seq_plus1 = args.seq_len + 1
    jobs = []
    shard_domains = {}
    for domain, paths in per_domain_files.items():
        budget = total * PHASE1_SHARES[domain]
        per_file = math.ceil(budget / max(1, len(paths)))
        for i, p in enumerate(paths):
            name = f"shard_{domain.lower()}_{i:05d}.npy"
            shard_domains[name] = domain
            jobs.append((p, domain, str(train_dir), tokenizer_dir, seq_plus1, name, per_file))

    print(f"=== tokenizing {len(jobs)} files with {args.workers} workers ===")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(tokenize_file, j): j[5] for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            name, rows, toks, domain = fut.result()
            if not name:
                continue
            results.append((name, rows, toks, domain))
            done = sum(r[2] for r in results)
            print(
                f"  [{k}/{len(jobs)}] {name} rows={rows} tokens={toks/1e6:.1f}M "
                f"| total {done/1e9:.2f}B in {time.time()-t0:.0f}s",
                flush=True,
            )

    # Interleave shards across domains so any prefix of the stream is mixed.
    by_dom: dict[str, list] = defaultdict(list)
    for r in sorted(results):
        by_dom[r[3]].append(r)
    order = []
    i = 0
    while any(len(v) > i for v in by_dom.values()):
        for d in ("English", "Chinese", "Code", "Math"):
            if len(by_dom.get(d, [])) > i:
                order.append(by_dom[d][i])
        i += 1

    # Carve the validation set off the tail of the interleaved stream.
    valid_rows_needed = int(args.valid_tokens // seq_plus1)
    valid_shards, train_shards = [], []
    acc = 0
    for name, rows, toks, domain in reversed(order):
        if acc < valid_rows_needed:
            src = train_dir / name
            arr = np.load(src, mmap_mode="r")
            n_take = min(rows, valid_rows_needed - acc)
            np.save(valid_dir / name, np.array(arr[:n_take]))
            valid_shards.append((name, n_take))
            acc += n_take
            if n_take < rows:
                np.save(src, np.array(arr[n_take:]))
                train_shards.append((name, rows - n_take))
            else:
                src.unlink()
        else:
            train_shards.append((name, rows))
    train_shards.reverse()

    mt = write_metadata(train_dir, train_shards)
    mv = write_metadata(valid_dir, valid_shards)

    tok_train = mt["size"] * seq_plus1
    tok_valid = mv["size"] * seq_plus1
    print("\n=== done ===")
    print(f"train: {mt['size']} samples  {tok_train/1e9:.2f}B tokens  {len(mt['file_list'])} shards -> {train_dir}")
    print(f"valid: {mv['size']} samples  {tok_valid/1e6:.1f}M tokens  {len(mv['file_list'])} shards -> {valid_dir}")
    dom_tok = defaultdict(int)
    for name, rows, toks, domain in results:
        dom_tok[domain] += toks
    tot = sum(dom_tok.values()) or 1
    print("mixture (target -> actual):")
    for d, share in PHASE1_SHARES.items():
        print(f"  {d:8s} {share*100:5.1f}% -> {dom_tok[d]/tot*100:5.1f}%  ({dom_tok[d]/1e9:.2f}B)")


if __name__ == "__main__":
    main()
