#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Sample 1,000 rows from each H3 cluster, compute SPECTER embeddings with
tokenizer chunking (win=100, stride=50, mean-pooled), and persist results
for fast reuse during visualization experiments.

Outputs (in OUTPUT_DIR):
  - blob_sample_embeddings.npy        (float16, shape [N, D])
  - blob_sample_metadata.parquet      (row-aligned to embeddings)
  - blob_sample_info.json             (summary: counts, timing, params)

Defaults:
  DB path:   C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\backup_2025_08_08\merged_works_labeled.db
  Table:     works_labeled
  Output:    C:\Users\Admin\Documents\Master-Thesis\results\topic_modeling\supercluster_umap_visulization
"""

import os
import json
import time
import argparse
import sqlite3
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer



# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────
DB_DEFAULT = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\full_dataset_new_labels\merged_works_labeled.db"
TABLE_DEFAULT = "works_labeled"
OUTPUT_DIR_DEFAULT = r"C:\Users\Admin\Documents\Master-Thesis\results\topic_modeling\supercluster_umap_visulization"

MODEL_NAME = "allenai-specter"
TOKENIZER_NAME = "allenai/specter"

SEED = 44
LANG = "en"
SAMPLE_PER_H3 = 1000         # up to this many per H3
WIN = 100
STRIDE = 50
EMB_DTYPE = np.float16        # compact, fast; use float32 if you need exact math later



# ──────────────────────────────────────────────────────────────────────────────
# Minimal text preprocess (consistent with your pipeline)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_text(title: str, cleaned_abstract: str) -> str:
    title = (title or "").strip()
    abst = (cleaned_abstract or "").strip()
    if not title and not abst:
        return ""
    return f"{title}. {abst}".strip()


# ──────────────────────────────────────────────────────────────────────────────
# Embedding with chunking (win/stride) and mean pooling (exactly as discussed)
# ──────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def embed_texts(texts: List[str], model, tokenizer, device: str, win: int = WIN, stride: int = STRIDE):
    embs = []
    for txt in texts:
        toks = tokenizer.tokenize(txt) if txt else []
        if not toks:
            v = model.encode([""], device=device)
            embs.append(v.mean(axis=0))
            continue
        chunks = [toks[i:i + win] for i in range(0, len(toks), stride)]
        ctext = [tokenizer.convert_tokens_to_string(c) for c in chunks] or [""]
        v = model.encode(ctext, device=device)  # (n_chunks, dim)
        embs.append(v.mean(axis=0))
    return np.vstack(embs)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Sample H3 clusters and precompute SPECTER embeddings.")
    ap.add_argument("--db-path", type=str, default=DB_DEFAULT)
    ap.add_argument("--table", type=str, default=TABLE_DEFAULT)
    ap.add_argument("--lang", type=str, default=LANG)
    ap.add_argument("--sample-per-h3", type=int, default=SAMPLE_PER_H3)
    ap.add_argument("--out-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=1024, help="Batch size for embedding compute")
    ap.add_argument("--fp16", action="store_true", default=True, help="Store embeddings as float16")
    args = ap.parse_args()

    np.random.seed(SEED)

    os.makedirs(args.out_dir, exist_ok=True)
    out_emb_path = os.path.join(args.out_dir, "blob_sample_embeddings_new_003.npy")
    out_meta_path = os.path.join(args.out_dir, "blob_sample_metadata_new_003.parquet")
    out_info_path = os.path.join(args.out_dir, "blob_sample_info_new_003.json")

    t0 = time.time()
    print("=" * 60)
    print("H3 CLUSTER SAMPLING + EMBEDDING (SPECTER, chunked mean-pooled)")
    print("=" * 60)

    # ── Step 1: Sample up to N per H3 (keeping H1/H2 in key)
    print("\n[1/4] Sampling rows per H3 cluster from SQLite")
    conn = sqlite3.connect(args.db_path)
    try:
        sampling_query = f"""
        WITH clustered AS (
            SELECT
                id, title, cleaned_abstract,
                h1_cluster, h2_cluster, h3_cluster,
                ROW_NUMBER() OVER (
                    PARTITION BY h1_cluster, h2_cluster, h3_cluster
                    ORDER BY RANDOM()
                ) AS rn
            FROM {args.table}
            WHERE language = ?
              AND cleaned_abstract IS NOT NULL
              AND h1_cluster IS NOT NULL
              AND h2_cluster IS NOT NULL
              AND h3_cluster IS NOT NULL
        )
        SELECT id, title, cleaned_abstract, h1_cluster, h2_cluster, h3_cluster
        FROM clustered
        WHERE rn <= ?
        ORDER BY h1_cluster, h2_cluster, h3_cluster, rn
        """
        df = pd.read_sql_query(sampling_query, conn, params=(args.lang, args.sample_per_h3))
    finally:
        conn.close()

    if df.empty:
        raise SystemExit("No rows sampled. Check DB path/table/language.")

    # Build globally unique H3 key (avoid local-ID collisions)
    df["h3_key"] = (
        df["h1_cluster"].astype(str) + "-" +
        df["h2_cluster"].astype(str) + "-" +
        df["h3_cluster"].astype(str)
    )

    # Report sampling sizes per H3
    counts = df.groupby("h3_key", observed=True).size().sort_values(ascending=True)
    unique_h3 = counts.shape[0]
    print(f"[i] Sampled {len(df):,} rows across {unique_h3} H3 groups.")
    print(f"[i] Per-H3 sample: min={counts.min()}, median={int(counts.median())}, max={counts.max()}")

    # ── Step 2: Load model+tokenizer
    print("\n[2/4] Loading SPECTER model and tokenizer")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # ── Step 3: Compute embeddings (chunked mean-pooled)
    print("\n[3/4] Computing embeddings")
    texts = [preprocess_text(t, a) for t, a in zip(df["title"].values, df["cleaned_abstract"].values)]

    # First batch to determine embedding dim
    bs = max(1, int(args.batch_size))
    first = texts[:min(bs, len(texts))]
    first_emb = embed_texts(first, model, tokenizer, device, WIN, STRIDE)
    emb_dim = int(first_emb.shape[1])
    dtype = np.float16 if args.fp16 else np.float32

    # Pre-allocate array
    N = len(texts)
    embs = np.empty((N, emb_dim), dtype=dtype)
    embs[:first_emb.shape[0], :] = first_emb.astype(dtype)

    processed = first_emb.shape[0]
    with tqdm(total=N, initial=processed, desc="Embedding", unit="rows") as pbar:
        while processed < N:
            end = min(processed + bs, N)
            batch = texts[processed:end]
            batch_emb = embed_texts(batch, model, tokenizer, device, WIN, STRIDE)
            embs[processed:end, :] = batch_emb.astype(dtype)
            processed = end
            pbar.update(len(batch))

    # ── Step 4: Persist outputs
    print("\n[4/4] Saving outputs")
    # Save embeddings
    np.save(out_emb_path, embs)
    # Save metadata (row-aligned)
    meta = pd.DataFrame({
        "paper_id": df["id"].astype(str).values,
        "title": df["title"].fillna("").astype(str).values,
        "abstract": df["cleaned_abstract"].fillna("").astype(str).values,
        "h1_cluster": df["h1_cluster"].values,
        "h2_cluster": df["h2_cluster"].values,
        "h3_cluster": df["h3_cluster"].values,
        "h3_key": df["h3_key"].values,
    })
    meta.to_parquet(out_meta_path, index=False)

    # Save run info
    info = {
        "db_path": args.db_path,
        "table": args.table,
        "language": args.lang,
        "sample_per_h3_target": args.sample_per_h3,
        "unique_h3_groups": int(unique_h3),
        "rows_total": int(N),
        "embedding_dim": int(emb_dim),
        "dtype": "float16" if args.fp16 else "float32",
        "device": device,
        "win": WIN,
        "stride": STRIDE,
        "model_name": MODEL_NAME,
        "tokenizer_name": TOKENIZER_NAME,
        "counts_per_h3": counts.to_dict(),
        "time_sec": round(time.time() - t0, 3),
        "outputs": {
            "embeddings_npy": out_emb_path,
            "metadata_parquet": out_meta_path,
            "info_json": out_info_path
        }
    }
    with open(out_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"[✓] Saved embeddings: {out_emb_path}  (shape={embs.shape}, dtype={embs.dtype})")
    print(f"[✓] Saved metadata:   {out_meta_path}  (rows={len(meta):,})")
    print(f"[✓] Saved summary:    {out_info_path}")
    print(f"[✓] Total time:       {info['time_sec']} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
