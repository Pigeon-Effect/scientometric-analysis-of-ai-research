"""
h3_clustering.py  – Hierarchical level-3 clustering with
 - user-selectable (h1, h2) subset
 - random 30 k-sample (or all if <30 k)
 - 30-cluster K-means
 - central/random-abstract export
 - interactive UMAP

Usage (examples)
----------------
# same machine paths as before
python h3_clustering.py --h1 0 --h2 0
python h3_clustering.py --h1 2 --h2 4 --sample 25000

Change the DB_PATH constant if your database moved.
"""

import os, json, re, sqlite3, argparse, random, numpy as np, pandas as pd, torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import umap.umap_ as umap
import plotly.express as px

# ───────────────────────────── Configuration ──────────────────────────────
DB_PATH = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\h1_cluster_subsets\engineering_dataset.db"
OUTPUT_ROOT = r"C:\Users\Admin\Documents\Master-Thesis\code\analysis\topic_modeling\sklearn_feature_extraction\h3_clustering\resources"
MODEL_NAME  = "allenai-specter"  # Fixed: use same as working script for SentenceTransformer
TOKENIZER_NAME = "allenai/specter"  # Separate name for AutoTokenizer
SAMPLE_CAP  = 30_000
N_CLUSTERS  = 30
SEED        = 42

# ── TF-IDF cluster naming configuration
REPRESENTATIVENESS_OF_TFIDF = 0.9  # 0.0 = fully distinctive, 1.0 = fully representative

# ── Hierarchical cluster configuration (change these to target different clusters)
H1_CLUSTER_ID = 4  # Set your target h1 cluster ID here
H2_CLUSTER_ID = 5  # Set your target h2 cluster ID here

# ── Auto-generated hierarchical naming
CLUSTER_HIERARCHY = f"h1_{H1_CLUSTER_ID:02d}_h2_{H2_CLUSTER_ID:02d}"
UMAP_FILENAME = f"umap_interactive_{CLUSTER_HIERARCHY}.html"
ABSTRACTS_FILENAME = f"cluster_abstracts_summary_{CLUSTER_HIERARCHY}.json"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ───────────────────────────── Helper functions ───────────────────────────
def preprocess(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(w for w in text.split() if len(w) > 2)

def tfidf_cluster_names(vectorizer, tfidf_matrix, labels, top_n=8, representativeness=0.7):
    """
    Generate cluster names based on most representative terms.

    Args:
        representativeness: 0.0 = distinctive terms (high IDF), 1.0 = representative terms (high TF in cluster)
    """
    feature_names = vectorizer.get_feature_names_out()
    n_clusters = labels.max() + 1

    print(f"[INFO] Generating cluster names with representativeness={representativeness:.1f}")
    print(f"[INFO] representativeness=0.0 → distinctive terms (rare globally)")
    print(f"[INFO] representativeness=1.0 → representative terms (frequent in cluster)")

    # Convert to dense for easier calculation
    tfidf_dense = tfidf_matrix.toarray()

    names = {}
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_docs = tfidf_dense[cluster_mask]

        if len(cluster_docs) == 0:
            names[cluster_id] = f"empty_cluster_{cluster_id}"
            continue

        if representativeness >= 0.5:
            # HIGH REPRESENTATIVENESS: Focus on terms that appear frequently in this cluster
            # Calculate mean term frequency within the cluster (ignoring IDF weighting)
            cluster_term_scores = cluster_docs.mean(axis=0)

            # Optional: Boost terms that are more unique to this cluster
            if representativeness < 1.0:
                # Calculate how much more frequent terms are in this cluster vs others
                other_mask = ~cluster_mask
                if np.sum(other_mask) > 0:
                    other_docs = tfidf_dense[other_mask]
                    global_mean = other_docs.mean(axis=0)
                    # Blend cluster frequency with distinctiveness
                    blend_factor = 2.0 * (1.0 - representativeness)
                    cluster_term_scores = cluster_term_scores + blend_factor * (cluster_term_scores - global_mean)
        else:
            # LOW REPRESENTATIVENESS: Use original TF-IDF (emphasizes rare terms)
            cluster_term_scores = cluster_docs.mean(axis=0)

        # Get top terms, filtering out noise
        top_indices = cluster_term_scores.argsort()[::-1]
        top_terms = []

        for idx in top_indices:
            term = feature_names[idx]
            score = cluster_term_scores[idx]

            # Filter out noise terms
            if (len(term) < 3 or                    # too short
                term.isdigit() or                   # just numbers
                score < 0.005 or                    # too low score
                any(char.isdigit() for char in term) or  # contains numbers
                term in ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'they', 'have', 'been', 'were', 'more', 'also', 'can', 'may', 'such', 'used', 'use', 'using', 'based', 'new', 'different', 'various', 'many', 'other', 'these', 'some', 'most', 'first', 'two', 'three', 'both', 'each', 'well', 'high', 'low', 'large', 'small']):
                continue

            top_terms.append(term)
            if len(top_terms) >= top_n:
                break

        # Create cluster name
        if top_terms:
            names[cluster_id] = " / ".join(top_terms)
        else:
            names[cluster_id] = f"cluster_{cluster_id}"

        print(f"[h3-{cluster_id:02d}] {names[cluster_id]}")

    return names

def export_summaries(df, emb, cluster_col, name_col, out_json, n_central=20, n_random=100):
    summaries = {}
    for cid in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == cid]
        idx   = sub.index.to_numpy()
        subemb= emb[idx]
        center= subemb.mean(axis=0)
        dist  = np.linalg.norm(subemb - center, axis=1)
        order = np.argsort(dist)
        central_idx = idx[order[: min(n_central, len(idx))]]
        rand_idx    = np.random.choice(idx, size=min(n_random, len(idx)), replace=False)
        summaries[str(cid)] = {
            "cluster_name"       : sub[name_col].iloc[0],
            "num_documents"      : len(sub),
            "most_central_abstracts": sub.loc[central_idx, ["title","cleaned_abstract"]].to_dict("records"),
            "random_abstracts"      : sub.loc[rand_idx,    ["title","cleaned_abstract"]].to_dict("records"),
        }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"[✓] Abstract summaries → {out_json}")

def save_umap(df, cluster_col, name_col, out_html):
    dfp = df.copy()
    dfp["label"] = dfp[cluster_col].astype(str) + ": " + dfp[name_col]

    # Sort clusters by ID for consistent legend ordering
    unique_clusters = sorted(dfp[cluster_col].unique())
    sorted_labels = []
    for cluster_id in unique_clusters:
        cluster_name = dfp[dfp[cluster_col] == cluster_id][name_col].iloc[0]
        sorted_labels.append(f"{cluster_id}: {cluster_name}")

    colors = px.colors.qualitative.Light24 * 3
    mapping = {lab: colors[i % len(colors)] for i, lab in enumerate(sorted_labels)}

    fig = px.scatter(
        dfp, x="x", y="y", color="label", hover_data={"label": True},
        color_discrete_map=mapping, height=900, width=1800,  # Increased width from 1200 to 1800
        title="h3 interactive UMAP",
        category_orders={"label": sorted_labels}  # Ensure legend is sorted by cluster ID
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.write_html(out_html)
    print(f"[✓] UMAP → {out_html}")

# ───────────────────────────── Main pipeline ───────────────────────────────
def main(args):
    # ── Use configured cluster IDs for consistent naming (override args if needed)
    h1_id = H1_CLUSTER_ID if H1_CLUSTER_ID is not None else args.h1
    h2_id = H2_CLUSTER_ID if H2_CLUSTER_ID is not None else args.h2

    # ── Generate hierarchical file names
    cluster_hierarchy = f"h1_{h1_id:02d}_h2_{h2_id:02d}"
    umap_filename = f"umap_interactive_{cluster_hierarchy}.html"
    abstracts_filename = f"cluster_abstracts_summary_{cluster_hierarchy}.json"

    # ── I/O paths with hierarchical naming
    out_dir = os.path.join(OUTPUT_ROOT, cluster_hierarchy)
    print(f"[DEBUG] Creating output directory: {out_dir}")

    try:
        os.makedirs(out_dir, exist_ok=True)
        print(f"[DEBUG] Directory created successfully: {os.path.exists(out_dir)}")
    except Exception as e:
        print(f"[ERROR] Failed to create directory {out_dir}: {e}")
        return

    umap_html = os.path.join(out_dir, umap_filename)
    json_path = os.path.join(out_dir, abstracts_filename)

    print(f"[DEBUG] JSON path will be: {json_path}")
    print(f"[DEBUG] UMAP path will be: {umap_html}")

    # ── Load subset
    conn = sqlite3.connect(DB_PATH)
    q = """SELECT title, cleaned_abstract, h1_cluster, h2_cluster
           FROM works_labeled
           WHERE cleaned_abstract IS NOT NULL AND language='en'
                 AND h1_cluster=? AND h2_cluster=?"""
    df = pd.read_sql_query(q, conn, params=(h1_id, h2_id))
    conn.close()
    print(f"[i] total docs in (h1={h1_id}, h2={h2_id}): {len(df):,}")

    if len(df) == 0:
        print("[!] subset empty – abort."); return

    # ── Sample
    sample_n = min(args.sample, len(df))
    df = df.sample(n=sample_n, random_state=SEED).reset_index(drop=True)
    print(f"[i] clustering on {len(df):,} documents")

    # ── Preprocess
    df["processed_text"] = (df["title"].fillna("") + ". " + df["cleaned_abstract"]).apply(preprocess)

    # ── Embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    win, stride = 100, 50
    embeds = []
    for txt in tqdm(df["processed_text"], desc="embeddings"):
        toks   = tokenizer.tokenize(txt)
        chunks = [toks[i:i+win] for i in range(0, len(toks), stride)]
        ctext  = [tokenizer.convert_tokens_to_string(c) for c in chunks] or ['']
        embeds.append(model.encode(ctext, device=device).mean(axis=0))
    embeds = np.vstack(embeds)

    # ── K-means h3
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)
    df["h3_cluster"] = km.fit_predict(embeds)

    # ── UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED)
    df[["x","y"]] = reducer.fit_transform(embeds)

    # ── Cluster names
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vec.fit_transform(df["processed_text"])
    print(f"[DEBUG] About to call tfidf_cluster_names with representativeness={REPRESENTATIVENESS_OF_TFIDF}")
    names = tfidf_cluster_names(vec, tfidf, df["h3_cluster"].values, representativeness=REPRESENTATIVENESS_OF_TFIDF)
    df["h3_name"] = df["h3_cluster"].map(names)

    # ── Export with hierarchical naming
    export_summaries(df, embeds, "h3_cluster", "h3_name", json_path)
    save_umap(df, "h3_cluster", "h3_name", umap_html)

    print(f"[✓] Files exported with hierarchy: {cluster_hierarchy}")
    print(f"    - UMAP: {umap_filename}")
    print(f"    - Abstracts: {abstracts_filename}")

# ────────────────────────────── CLI entry ──────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h1", type=int, default=H1_CLUSTER_ID, help=f"parent h1_cluster id (default: {H1_CLUSTER_ID})")
    p.add_argument("--h2", type=int, default=H2_CLUSTER_ID, help=f"child h2_cluster id (default: {H2_CLUSTER_ID})")
    p.add_argument("--sample", type=int, default=SAMPLE_CAP, help="max docs to cluster (default 30 k)")
    main(p.parse_args())
