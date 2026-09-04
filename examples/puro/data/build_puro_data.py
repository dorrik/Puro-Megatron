#!/usr/bin/env python3
"""Build a PROM packed-NPY dataset for Puro-Megatron from the released Puro-2B corpus.

The training code (`CustomGPTDataset` in pretrain_gpt.py) expects a directory of

    metadata.json   {"size": N, "file_list": [...], "file_len": [...]}
    shard_00000.npy  int32 array of shape (rows, seq_len + 1)

where every row is a contiguous slice of a token stream in which documents are
separated by the EOS id. `--attention-type thd` reconstructs packed-sequence
metadata from those EOS positions at run time, so plain concatenation is the
correct layout.

Source: https://huggingface.co/datasets/thu-pacman/Puro-2B, tokenized with the
bundled `qwen2_tokenizer/`. Domain shares default to the published mixture.

Runs in three stages so the work fits a Slurm job array and never needs the
whole corpus on disk at once:

    plan      -> manifest.json assigning a token budget to each source file
    run       -> one array task per slice: download, tokenize, DELETE parquet
    finalize  -> merge per-task results into train/ and valid/ metadata.json

Peak parquet on disk is therefore (concurrent tasks) x (one file), not the
866 GB of the full corpus.

    python build_puro_data.py plan     --out-dir DIR --target-tokens 4.39e11
    python build_puro_data.py run      --out-dir DIR --task-id $SLURM_ARRAY_TASK_ID \
                                       --num-tasks $SLURM_ARRAY_TASK_COUNT
    python build_puro_data.py finalize --out-dir DIR
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
from huggingface_hub import hf_hub_download, list_repo_tree

REPO_ID = "thu-pacman/Puro-2B"
TOKENIZER_SUBDIR = "qwen2_tokenizer"
EOS_ID = 151645  # <|im_end|>, matches --eos-token-id in the Puro recipes
VOCAB_SIZE = 151936

# Published domain mixtures (dataset card "Dataset Summary").
SHARES = {
    "phase1": {"English": 0.732, "Chinese": 0.117, "Code": 0.079, "Math": 0.072},
    "phase2": {"English": 0.594, "Chinese": 0.094, "Code": 0.115, "Math": 0.183},
}

# Published per-domain token totals (dataset card). Used to calibrate a
# tokens-per-byte ratio SEPARATELY PER DOMAIN: the parquet is compressed and
# the ratio varies ~3x across domains (English prose compresses far better than
# Chinese UTF-8), so a single corpus-wide ratio mis-plans every domain.
DOMAIN_TOKENS = {
    "phase1": {"English": 321e9, "Chinese": 51.3e9, "Code": 34.6e9, "Math": 31.6e9},
    "phase2": {"English": 558e9, "Chinese": 88.5e9, "Code": 108e9, "Math": 171e9},
}

COMPONENT_DOMAIN = {
    "fineweb-edu-dedup": "English",
    "cosmopedia-v2": "English",
    "arxiv": "English",
    "nemotron-high-quality": "English",
    "nemotron-high-quality-synthetic": "English",
    "dclm": "English",
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
    "megamath-code": "Code",
    "finemath": "Math",
    "swallow-math-v2": "Math",
    "nemotron-cc-math-v1-4plus": "Math",
    "megamath-web-pro": "Math",
}

# Bounded so a worker's peak resident set stays ~250 MB: one shard of output
# plus one read batch. An earlier version accumulated 150M tokens per worker
# and was OOM-killed at 48 workers against a 160 GB limit.
SHARD_TOKENS = 50_000_000
READ_BATCH_ROWS = 256

_TOKENIZER = None


def get_tokenizer(tokenizer_dir: str):
    global _TOKENIZER
    if _TOKENIZER is None:
        from tokenizers import Tokenizer

        _TOKENIZER = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
    return _TOKENIZER


def fetch_tokenizer(out: Path) -> str:
    d = out / "tokenizer"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "tokenizer.json").exists():
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            p = hf_hub_download(REPO_ID, f"{TOKENIZER_SUBDIR}/{f}", repo_type="dataset")
            shutil.copy(p, d / f)
    return str(d)


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
def stage_plan(args):
    """Assign a token budget to each source file without downloading any of them.

    Reading `token_count` remotely costs ~40 s/file (HTTP range reads over the
    footer plus a column), which is 80+ minutes across the corpus. File sizes
    come from the repo tree for free, and the corpus-wide tokens-per-byte ratio
    is accurate enough to choose files; the exact budget is enforced during
    tokenization anyway.
    """
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shares = SHARES[args.phase]

    entries = [
        e for e in list_repo_tree(REPO_ID, args.phase, repo_type="dataset", recursive=True)
        if getattr(e, "size", None) and e.path.endswith(".parquet")
    ]
    by_component: dict[str, list] = defaultdict(list)
    for e in entries:
        by_component[e.path.split("/")[1]].append(e)
    for v in by_component.values():
        v.sort(key=lambda e: e.path)

    domain_bytes: dict[str, int] = defaultdict(int)
    for e in entries:
        d = COMPONENT_DOMAIN.get(e.path.split("/")[1])
        if d:
            domain_bytes[d] += e.size
    tok_per_byte = {
        d: DOMAIN_TOKENS[args.phase][d] / domain_bytes[d]
        for d in shares if domain_bytes.get(d)
    }
    print(f"{args.phase}: {sum(e.size for e in entries)/1e9:.1f} GB on disk")
    for d, r in tok_per_byte.items():
        print(f"  {d:8s} {domain_bytes[d]/1e9:7.1f} GB -> "
              f"{DOMAIN_TOKENS[args.phase][d]/1e9:6.1f}B tokens  ({r:.3f} tok/byte)")

    jobs = []
    for domain, share in shares.items():
        comps = sorted(c for c, d in COMPONENT_DOMAIN.items() if d == domain and by_component.get(c))
        if not comps:
            print(f"  !! no components for {domain}", file=sys.stderr)
            continue
        budget = args.target_tokens * share
        remaining = budget
        idx = 0
        while remaining > 0 and any(idx < len(by_component[c]) for c in comps):
            for c in comps:
                if remaining <= 0 or idx >= len(by_component[c]):
                    continue
                e = by_component[c][idx]
                take = min(e.size * tok_per_byte[domain], remaining)
                remaining -= take
                jobs.append({"path": e.path, "domain": domain,
                             "budget": int(take), "bytes": e.size})
            idx += 1
        if remaining > budget * 0.02:
            print(f"  !! {domain}: {remaining/1e9:.2f}B short of {budget/1e9:.2f}B target",
                  file=sys.stderr)

    got: dict[str, float] = defaultdict(float)
    for j in jobs:
        got[j["domain"]] += j["budget"]
    manifest = {"phase": args.phase, "seq_len": args.seq_len,
                "target_tokens": args.target_tokens, "jobs": jobs}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nplanned {len(jobs)} files, {sum(j['bytes'] for j in jobs)/1e9:.1f} GB to download")
    for d, share in shares.items():
        print(f"  {d:8s} target {share*100:5.1f}%  planned {got[d]/1e9:7.2f}B")
    print(f"manifest -> {out/'manifest.json'}")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def tokenize_chunk(spec) -> list[dict]:
    """Tokenize one row-group range of an already-downloaded parquet.

    The unit of work is a row-group range, not a file: phase1 parts carry up to
    ~4.9B tokens, so one-worker-per-file leaves a single process running for
    hours while the rest of the pool idles.
    """
    (local, domain, budget, out_dir, tokenizer_dir, seq_plus1, tag,
     rg_start, rg_end) = spec
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    tok = get_tokenizer(tokenizer_dir)

    produced = 0
    results: list[dict] = []
    pending: list[np.ndarray] = []
    pending_tokens = 0
    leftover = np.empty(0, dtype=np.int32)
    shard_i = 0

    def flush():
        nonlocal pending, pending_tokens, shard_i
        if not pending:
            return
        arr = np.concatenate(pending, axis=0)
        assert arr.max() < VOCAB_SIZE, f"token id {arr.max()} >= vocab {VOCAB_SIZE}"
        name = f"shard_{tag}_{shard_i:04d}.npy"
        np.save(Path(out_dir) / name, arr)
        results.append({"shard": name, "rows": int(arr.shape[0]),
                        "tokens": int(arr.size), "domain": domain})
        shard_i += 1
        pending, pending_tokens = [], 0

    pf = pq.ParquetFile(local)
    rgs = list(range(rg_start, min(rg_end, pf.num_row_groups)))
    if not rgs:
        return results
    for batch in pf.iter_batches(batch_size=READ_BATCH_ROWS, columns=["text"],
                                 row_groups=rgs):
        texts = batch.column("text").to_pylist()
        encs = (tok.encode_batch_fast(texts) if hasattr(tok, "encode_batch_fast")
                else tok.encode_batch(texts))
        parts = [leftover]
        for e in encs:
            parts.append(np.asarray(e.ids, dtype=np.int32))
            parts.append(np.array([EOS_ID], dtype=np.int32))
        del encs, texts
        buf = np.concatenate(parts)
        n_full = len(buf) // seq_plus1
        if n_full:
            block = buf[: n_full * seq_plus1].reshape(n_full, seq_plus1)
            pending.append(block)
            pending_tokens += block.size
            produced += block.size
        leftover = buf[n_full * seq_plus1:].copy()
        del buf, parts
        if pending_tokens >= SHARD_TOKENS:
            flush()
        if produced >= budget:
            break
    flush()
    return results


def stage_run(args):
    out = Path(args.out_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    jobs = manifest["jobs"]
    seq_plus1 = manifest["seq_len"] + 1

    mine = [(i, j) for i, j in enumerate(jobs) if i % args.num_tasks == args.task_id]
    if not mine:
        print(f"task {args.task_id}: nothing to do")
        return
    shard_dir = out / "shards"
    cache = out / "_parquet" / f"task{args.task_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = fetch_tokenizer(out)

    print(f"task {args.task_id}/{args.num_tasks}: {len(specs)} source files, "
          f"{sum(j['budget'] for _, j in mine)/1e9:.2f}B tokens, {args.workers} workers",
          flush=True)

    produced: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, (i, j) in enumerate(mine, 1):
            try:
                local = hf_hub_download(REPO_ID, j["path"], repo_type="dataset",
                                        local_dir=str(cache))
            except Exception as e:
                print(f"  !! download {j['path']}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                continue
            try:
                n_rg = pq.ParquetFile(local).num_row_groups
                n_chunks = max(1, min(n_rg, args.workers))
                per_rg = math.ceil(n_rg / n_chunks)
                per_budget = math.ceil(j["budget"] / n_chunks)
                specs = []
                for c in range(n_chunks):
                    lo, hi = c * per_rg, min((c + 1) * per_rg, n_rg)
                    if lo >= hi:
                        break
                    specs.append((local, j["domain"], per_budget, str(shard_dir),
                                  tokenizer_dir, seq_plus1,
                                  f"{j['domain'].lower()}_{i:04d}_{c:03d}", lo, hi))
                got = 0
                for fut in as_completed([ex.submit(tokenize_chunk, sp) for sp in specs]):
                    try:
                        res = fut.result()
                    except Exception as e:
                        print(f"  !! chunk of {j['path']}: {type(e).__name__}: {e}",
                              file=sys.stderr, flush=True)
                        continue
                    produced.extend(res)
                    got += sum(r["tokens"] for r in res)
                done = sum(r["tokens"] for r in produced)
                print(f"  [{n}/{len(mine)}] {j['path']} -> {got/1e9:.2f}B "
                      f"({len(specs)} chunks) | total {done/1e9:.2f}B "
                      f"in {time.time()-t0:.0f}s", flush=True)
            finally:
                # Delete as we go: peak parquet on disk stays at one file per task.
                if not args.keep_parquet:
                    try:
                        os.remove(local)
                    except OSError:
                        pass

    (out / f"task_{args.task_id:04d}.json").write_text(json.dumps(produced, indent=2))
    shutil.rmtree(cache, ignore_errors=True)
    print(f"task {args.task_id}: {sum(r['tokens'] for r in produced)/1e9:.2f}B tokens")


# --------------------------------------------------------------------------
# finalize
# --------------------------------------------------------------------------
def stage_finalize(args):
    out = Path(args.out_dir)
    shard_dir = out / "shards"
    manifest = json.loads((out / "manifest.json").read_text())
    seq_plus1 = manifest["seq_len"] + 1

    results: list[dict] = []
    for f in sorted(out.glob("task_*.json")):
        results.extend(json.loads(f.read_text()))
    if not results:
        sys.exit("no task results found; did the array run?")
    present = {p.name for p in shard_dir.glob("*.npy")}
    results = [r for r in results if r["shard"] in present]

    by_dom: dict[str, list] = defaultdict(list)
    for r in sorted(results, key=lambda r: r["shard"]):
        by_dom[r["domain"]].append(r)
    # Interleave so any prefix of the stream carries the full mixture.
    order: list[dict] = []
    i = 0
    while any(len(v) > i for v in by_dom.values()):
        for d in ("English", "Chinese", "Code", "Math"):
            if len(by_dom.get(d, [])) > i:
                order.append(by_dom[d][i])
        i += 1

    train_dir, valid_dir = out / "train", out / "valid"
    for d in (train_dir, valid_dir):
        d.mkdir(parents=True, exist_ok=True)

    need_valid = int(args.valid_tokens // seq_plus1)
    valid: list[tuple[str, int]] = []
    train: list[tuple[str, int]] = []
    acc = 0
    for r in reversed(order):
        src = shard_dir / r["shard"]
        if acc < need_valid:
            arr = np.load(src, mmap_mode="r")
            take = min(r["rows"], need_valid - acc)
            np.save(valid_dir / r["shard"], np.array(arr[:take]))
            valid.append((r["shard"], take))
            acc += take
            if take < r["rows"]:
                np.save(train_dir / r["shard"], np.array(arr[take:]))
                train.append((r["shard"], r["rows"] - take))
            del arr
            src.unlink()
        else:
            os.replace(src, train_dir / r["shard"])
            train.append((r["shard"], r["rows"]))
    train.reverse()

    def write_meta(d: Path, shards):
        meta = {"size": sum(n for _, n in shards),
                "file_list": [s for s, _ in shards],
                "file_len": [n for _, n in shards],
                "sample_index_offset": 0}
        (d / "metadata.json").write_text(json.dumps(meta, indent=2))
        return meta

    mt = write_meta(train_dir, train)
    mv = write_meta(valid_dir, valid)
    dom: dict[str, int] = defaultdict(int)
    for r in results:
        dom[r["domain"]] += r["tokens"]
    tot = sum(dom.values()) or 1
    print(f"train: {mt['size']} samples  {mt['size']*seq_plus1/1e9:.2f}B tokens  "
          f"{len(mt['file_list'])} shards")
    print(f"valid: {mv['size']} samples  {mv['size']*seq_plus1/1e6:.1f}M tokens  "
          f"{len(mv['file_list'])} shards")
    print("mixture:")
    for d, share in SHARES[manifest["phase"]].items():
        print(f"  {d:8s} target {share*100:5.1f}%  actual {dom[d]/tot*100:5.1f}%  ({dom[d]/1e9:.2f}B)")
    shutil.rmtree(out / "_parquet", ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "run", "finalize"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--phase", default="phase1", choices=["phase1", "phase2"])
    ap.add_argument("--target-tokens", type=float, default=439e9)
    ap.add_argument("--valid-tokens", type=float, default=4e7)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-parquet", action="store_true",
                    help="do not delete each parquet after tokenizing it")
    args = ap.parse_args()
    {"plan": stage_plan, "run": stage_run, "finalize": stage_finalize}[args.stage](args)


if __name__ == "__main__":
    main()
