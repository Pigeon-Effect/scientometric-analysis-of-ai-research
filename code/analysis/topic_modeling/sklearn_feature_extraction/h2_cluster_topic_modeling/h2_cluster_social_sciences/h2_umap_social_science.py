"""
Social‑science macro‑mapping and UMAP visualisation
--------------------------------------------------
Key differences from the natural‑science script:
1.  Paths point to the SOCIAL‑SCIENCE curated‑cluster JSON and SQLite sample.
2.  Remap dictionary now reflects the SIX agreed macro‑clusters.
3.  Run‑time duplicate check ensures every original cluster ID appears
    exactly once; otherwise the script aborts.
4.  Colour palette condensed to those 6 clusters.
"""
import os, json, re, sqlite3, torch
import numpy as np, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── File paths ────────────────────────────────────────────────────────────────
JSON_PATH = r"C:\Users\Admin\Documents\Master-Thesis\code\analysis\topic_modeling\sklearn_feature_extraction\h2_cluster_engineering\resources\cluster_abstracts_summary.json"
DB_PATH   = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\h1_cluster_subsets\engineering_sample.db"
SVG_PATH  = r"C:\Users\Admin\Documents\Master-Thesis\code\analysis\topic_modeling\sklearn_feature_extraction\h2_cluster_engineering\resources\mapped_umap.svg"

# ── 1. Load curated clusters (reference docs) ─────────────────────────────────
with open(JSON_PATH, encoding="utf-8") as f:
    curated = json.load(f)

ref_texts, ref_labels = [], []
for cid, entry in curated.items():
    lbl = f"cluster_{cid}" if cid.isdigit() else cid
    for section in ("most_central_abstracts", "random_abstracts"):
        for item in entry[section]:
            title = (item.get("title") or "").strip()
            abst  = (item.get("cleaned_abstract") or "").strip()
            ref_texts.append(f"{title}. {abst}")
            ref_labels.append(lbl)

# ── 2. Embedding model & helper ───────────────────────────────────────────────
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = SentenceTransformer("allenai-specter", device=device)
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

def embed_sliding(texts, win=100, stride=50):
    vecs = []
    for txt in tqdm(texts, desc="Embedding"):
        toks   = tokenizer.tokenize(txt)
        chunks = [toks[i:i+win] for i in range(0, len(toks), stride)]
        chunk_txts = [tokenizer.convert_tokens_to_string(c) for c in chunks if c]
        emb = model.encode(chunk_txts or [""], device=device)
        vecs.append(emb.mean(axis=0))
    return np.vstack(vecs)

ref_emb = embed_sliding(ref_texts)

# ── 3. Load engineering subsample ─────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql_query("""
        SELECT id, title, cleaned_abstract
        FROM   works_labeled
        WHERE  cleaned_abstract IS NOT NULL
          AND language = 'en';
       """, conn)
conn.close()

# ── 4. Light preprocessing & embedding ────────────────────────────────────────
def clean(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r'[^a-z\s]', ' ', txt)
    return ' '.join(w for w in txt.split() if len(w) > 2)

df["processed_text"] = (df["title"].fillna('') + '. ' +
                        df["cleaned_abstract"].fillna('')).apply(clean)

abs_emb = embed_sliding(df["processed_text"].tolist())

# ── 5. Cosine‑similarity assignment to nearest reference doc ──────────────────
sim      = cosine_similarity(abs_emb, ref_emb)
nearest  = np.argmax(sim, axis=1)
df["ref_label"] = [ref_labels[i] for i in nearest]

# ── 6. Outlier detection (Isolation Forest) ───────────────────────────────────
print(f"Original dataset size: {len(df)}")

scaler          = StandardScaler()
abs_emb_scaled  = scaler.fit_transform(abs_emb)
iso_forest      = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
outlier_labels  = iso_forest.fit_predict(abs_emb_scaled)

mask            = outlier_labels == 1   # keep inliers
df              = df[mask].reset_index(drop=True)
abs_emb         = abs_emb[mask]

print(f"Filtered dataset size: {len(df)}  (removed {np.sum(outlier_labels==-1)} outliers)")

# ── 7. Macro‑area mapping ─────────────────────────────────────────────────────
remap = {
    # 0 – Cybersecurity & Privacy
    **{str(i): "Cybersecurity & Privacy"
       for i in (6,14,18,22,30,31,35,38,39,40,46,49,48)},

    # 1 – IoT & Edge Computing
    **{str(i): "IoT & Edge Computing"
       for i in (3,12,16,20,28,36,4,34,13,33,45)},

    # 2 – Anomaly & Fault Detection
    **{str(i): "Anomaly & Fault Detection"
       for i in (2,8,23,24)},

    # 3 – Energy Technologies
    **{str(i): "Energy Technologies"
       for i in (1,10,17,19,25,37)},

    # 4 – Autonomous Mobility
    **{str(i): "Autonomous Mobility"
       for i in (0,11,26,27,32,42,43,47)},

    # 5 – Robotics & Mechatronics
    **{str(i): "Robotics & Mechatronics"
       for i in (21,41,44,5,7,9,15,29)}
}

# ── 7a. Duplicate‑key sanity check --------------------------------------------
dupes = [k for k in set(remap) if list(remap.keys()).count(k) > 1]
if dupes or len(remap) != 50:
    missing = set(str(i) for i in range(50)) - set(remap)
    raise ValueError(f"Mapping error. Duplicates: {dupes}, Missing: {missing}")

df["macro_cluster"] = [
    remap.get(lbl.split('_')[-1], "Unmapped") for lbl in df["ref_label"]
]

# ── 8. 2‑D UMAP projection ────────────────────────────────────────────────────
umap_xy = umap.UMAP(n_neighbors=15, min_dist=0.1,
                    metric='cosine', random_state=42).fit_transform(abs_emb)
df["x"], df["y"] = umap_xy[:,0], umap_xy[:,1]

# ── 9. Plot -------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
palette = {
    "Cybersecurity & Privacy":      "#3a86ff",
    "IoT & Edge Computing":         "#ffbe0b",
    "Anomaly & Fault Detection":    "#7209b7",
    "Energy Technologies":          "#fb5607",
    "Autonomous Mobility":          "#2dd4bf",
    "Robotics & Mechatronics":      "#8b0000"
}

fig, ax = plt.subplots(figsize=(13, 15))
order = df["macro_cluster"].value_counts().index.tolist()

for cl in order:
    sub   = df[df["macro_cluster"] == cl]
    color = palette.get(cl, "#d3d3d3")
    ax.scatter(sub["x"], sub["y"], s=5, alpha=0.6, color=color)

# Legend with counts
handles, labels = [], []
for cl in order:
    handles.append(ax.scatter([], [], s=50, color=palette.get(cl, "#d3d3d3")))
    labels.append(f"{cl} ({len(df[df['macro_cluster']==cl])})")

ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.legend(handles, labels, loc='lower left',
          fontsize=12, framealpha=0.9)
plt.tight_layout()
plt.savefig(SVG_PATH, format="svg", bbox_inches="tight")
plt.close()

print("✓ UMAP written to:", SVG_PATH)
