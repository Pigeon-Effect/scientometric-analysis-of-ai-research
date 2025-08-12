# h3_cluster_labeling.py – Propagate fine → super labels inside one (h1,h2) slice
# -----------------------------------------------------------------------------
# Pipeline:
#   1. Open the reference JSON that was exported by `h3_clustering.py`, e.g.
#        .../resources/h1_00_h2_00/cluster_abstracts_summary_h1_00_h2_00.json
#   2. Embed every reference abstract with allenai‑specter (sliding‑window mean).
#   3. Pull **only** the documents that belong to that (h1,h2) pair from the
#      SQLite DB and embed them the same way.
#   4. Assign the nearest reference (cosine-similarity argmax) ⇒ fine_h3_id.
#   5. Collapse the 30 fine IDs into 5 thematic super‑clusters via H3_MAP.
#   6. Update the `works_labeled` table, setting `h3_cluster` for those rows.
#
# Example:
#     python h3_cluster_labeling.py --h1 0 --h2 0
# -----------------------------------------------------------------------------

import os, json, re, sqlite3, argparse, torch
import numpy as np, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────────────────── Configuration ──────────────────────────────
DB_PATH        = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\h1_cluster_subsets\engineering_dataset.db"
RES_ROOT       = r"C:\Users\Admin\Documents\Master-Thesis\code\analysis\topic_modeling\sklearn_feature_extraction\h3_clustering\resources"
MODEL_NAME     = "allenai-specter"
TOKENIZER_NAME = "allenai/specter"

# Default slice (can be overridden via CLI)
H1_CLUSTER_ID = 4
H2_CLUSTER_ID = 5

# ── Super‑cluster remapping (fine h3_id → macro code 0‑4) ──────────────────
H3_MAP = {
    # 0 – Autonomous Navigation & Path Planning
    **{str(i): 0 for i in (22, 26, 12, 3, 6, 5)},

    # 1 – Embodied Robotics
    **{str(i): 1 for i in (17, 20, 27, 28, 4)},

    # 2 – Task Scheduling & Decision Making
    **{str(i): 2 for i in (1, 8, 25, 14, 21, 15, 10)},

    # 3 – Control Systems & Fuzzy Logic
    **{str(i): 3 for i in (16, 7, 13, 29, 18, 0, 24, 11, 2, 19, 9, 23)},
}





if len(H3_MAP) != 30:
    raise ValueError("H3_MAP must cover exactly the 30 fine h3 cluster IDs (0‑29)")

# ───────────────────────────── Helper functions ───────────────────────────

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(w for w in text.split() if len(w) > 2)


def embed_sliding(texts, model, tokenizer, win=100, stride=50, device="cpu"):
    vecs = []
    for txt in tqdm(texts, desc="embedding"):
        toks   = tokenizer.tokenize(txt)
        chunks = [toks[i:i+win] for i in range(0, len(toks), stride)]
        ctexts = [tokenizer.convert_tokens_to_string(c) for c in chunks] or [""]
        emb    = model.encode(ctexts, device=device)
        vecs.append(emb.mean(axis=0))
    return np.vstack(vecs)

# ───────────────────────────── Core pipeline ──────────────────────────────

def main(args):
    h1_id = args.h1
    h2_id = args.h2

    # ------------------------------------------------------------------
    # 1. Build path to reference JSON produced by h3_clustering.py
    #    Example for (0,0):
    #    resources/h1_00_h2_00/cluster_abstracts_summary_h1_00_h2_00.json
    # ------------------------------------------------------------------
    cluster_hierarchy_id = f"h1_{h1_id:02d}_h2_{h2_id:02d}"
    json_name = f"cluster_abstracts_summary_{cluster_hierarchy_id}.json"
    json_path = os.path.join(RES_ROOT, cluster_hierarchy_id, json_name)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Reference abstracts not found: {json_path}")
    print(f"[i] reference JSON → {json_path}")

    # ── 2. Load curated abstracts ----------------------------------------
    with open(json_path, encoding="utf-8") as f:
        ref_json = json.load(f)

    ref_texts, ref_labels = [], []
    for cid, entry in ref_json.items():
        for section in ("most_central_abstracts", "random_abstracts"):
            for item in entry[section]:
                title = (item.get("title") or "").strip()
                abst  = (item.get("cleaned_abstract") or "").strip()
                ref_texts.append(f"{title}. {abst}")
                ref_labels.append(cid)  # cid is the fine h3 ID as string

    # ── 3. Embed reference set -------------------------------------------
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = SentenceTransformer(MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    ref_emb = embed_sliding(ref_texts, model, tokenizer, device=device)

    # ── 4. Pull subset from DB -------------------------------------------
    conn = sqlite3.connect(DB_PATH)
    q = """SELECT rowid, title, cleaned_abstract
             FROM works_labeled
             WHERE cleaned_abstract IS NOT NULL AND language='en'
                   AND h1_cluster=? AND h2_cluster=?"""
    subset = pd.read_sql_query(q, conn, params=(h1_id, h2_id))

    if subset.empty:
        print("[!] selected (h1,h2) slice is empty – nothing to label.")
        conn.close(); return
    print(f"[i] documents to label: {len(subset):,}")

    # ── 5. Preprocess + embed subset -------------------------------------
    subset["proc_txt"] = (subset["title"].fillna("") + ". " +
                            subset["cleaned_abstract"].fillna(""))\
                            .apply(preprocess)
    sub_emb = embed_sliding(subset["proc_txt"].tolist(), model, tokenizer, device=device)

    # ── 6. Nearest‑neighbour labelling -----------------------------------
    sim   = cosine_similarity(sub_emb, ref_emb)
    best  = np.argmax(sim, axis=1)
    subset["fine_h3_id"] = [ref_labels[i] for i in best]

    # ── 7. Collapse to super‑cluster codes --------------------------------
    subset["h3_cluster"] = [H3_MAP[fid] for fid in subset["fine_h3_id"]]

    # ── 8. Persist to DB --------------------------------------------------
    # Check if h3_cluster column exists, add it if not
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(works_labeled)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'h3_cluster' not in columns:
        conn.execute("ALTER TABLE works_labeled ADD COLUMN h3_cluster INTEGER;")
    conn.commit()

    # ── Security check: Prevent overwriting existing h3_cluster values ──
    rowids_to_update = subset["rowid"].tolist()

    # Process in batches to avoid "too many SQL variables" error
    batch_size = 999  # SQLite's default SQLITE_MAX_VARIABLE_NUMBER
    existing_clusters = []

    for i in range(0, len(rowids_to_update), batch_size):
        batch_rowids = rowids_to_update[i:i + batch_size]
        placeholders = ",".join("?" * len(batch_rowids))
        check_query = f"""SELECT rowid, h3_cluster 
                          FROM works_labeled 
                          WHERE rowid IN ({placeholders}) 
                          AND h3_cluster IS NOT NULL 
                          AND h3_cluster != ''
                          AND TYPEOF(h3_cluster) = 'integer'"""

        batch_existing = cursor.execute(check_query, batch_rowids).fetchall()
        existing_clusters.extend(batch_existing)

    if existing_clusters:
        conn.close()
        print(f"[!] ERROR: Cannot overwrite existing h3_cluster values!")
        print(f"[!] Found {len(existing_clusters)} rows that already have h3_cluster values:")
        for rowid, existing_value in existing_clusters[:5]:  # Show first 5 examples
            print(f"    rowid={rowid} already has h3_cluster={existing_value}")
        if len(existing_clusters) > 5:
            print(f"    ... and {len(existing_clusters) - 5} more rows")
        print(f"[!] Operation aborted. Please check your h1/h2 cluster selection.")
        print(f"[!] Only NULL/empty h3_cluster values can be overwritten.")
        return

    # Proceed with updates only if no existing values found
    # Use batch updates for better performance and accurate counting
    total_updated = 0
    batch_size = 999

    updates_data = [(int(h3c), int(rowid)) for rowid, h3c in subset[["rowid", "h3_cluster"]].itertuples(index=False)]

    for i in range(0, len(updates_data), batch_size):
        batch_data = updates_data[i:i + batch_size]

        # Use executemany for batch updates
        cursor.executemany("UPDATE works_labeled SET h3_cluster=? WHERE rowid=?;", batch_data)
        total_updated += cursor.rowcount if cursor.rowcount > 0 else len(batch_data)

    conn.commit(); conn.close()

    print(f"[✓] 'h3_cluster' updated for {total_updated:,} rows in slice (h1={h1_id}, h2={h2_id}).")

# ────────────────────────────── CLI entry ─────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h1", type=int, default=H1_CLUSTER_ID,
                   help=f"parent h1_cluster id (default: {H1_CLUSTER_ID})")
    p.add_argument("--h2", type=int, default=H2_CLUSTER_ID,
                   help=f"child h2_cluster id (default: {H2_CLUSTER_ID})")
    main(p.parse_args())
