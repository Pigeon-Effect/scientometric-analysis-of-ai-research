"""
h2_cluster_compsci.py
────────────────────────────────────────────────────────────────────────────
Adds a second-hierarchy label to every English record in the COMPUTER-SCIENCE
& MATHEMATICS sample and produces a 2-D UMAP visualisation.

Macro-clusters (codes 0-7)

    0 : Computer Vision & Image Processing
    1 : Natural Language Processing
    2 : Speech Recognition & Audio Processing
    3 : Neuromorphic Hardware Accelerators
    4 : Machine Learning Foundations
    5 : Biomarkers for Health & Performance
    6 : Ethical & Creative AI
    7 : Recommendation Systems
"""


import os, json, re, sqlite3, torch
import numpy as np, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns

# ── File & folder locations ───────────────────────────────────────────────────
ABSTRACT_JSON_PATH = r"C:\Users\Admin\Documents\Master-Thesis\code\analysis\topic_modeling\sklearn_feature_extraction\h2_cluster_computer_science\resources\cluster_abstracts_summary.json"
DB_PATH   = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\h1_cluster_subsets\computer_science_sample.db"
OUTPUT_DIR  = r"C:\Users\Admin\Documents\Master-Thesis\code\analysis\topic_modeling\sklearn_feature_extraction\h2_cluster_computer_science\resources"
SVG_PATH = os.path.join(OUTPUT_DIR, "computer_science_mapped_umap.svg")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load curated reference abstracts ---------------------------------------
with open(ABSTRACT_JSON_PATH, encoding="utf-8") as f:
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

# ── 2. Embed reference set -----------------------------------------------------
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = SentenceTransformer("allenai-specter", device=device)
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

def embed_sliding(texts, win=100, stride=50):
    vecs = []
    for txt in tqdm(texts, desc="Embedding reference docs"):
        toks   = tokenizer.tokenize(txt)
        chunks = [toks[i:i+win] for i in range(0, len(toks), stride)]
        chunk_txts = [tokenizer.convert_tokens_to_string(c) for c in chunks if c]
        emb = model.encode(chunk_txts or [""], device=device)
        vecs.append(emb.mean(axis=0))
    return np.vstack(vecs)

ref_emb = embed_sliding(ref_texts)

# ── 3. Load engineering sample -------------------------------------------------
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql_query("""
        SELECT id, title, cleaned_abstract
        FROM   works_labeled
        WHERE  cleaned_abstract IS NOT NULL
          AND language = 'en';
       """, conn)
conn.close()

# ── 4. Preprocess & embed corpus ----------------------------------------------
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(w for w in text.split() if len(w) > 2)

df["processed_text"] = (df["title"].fillna("") + ". " +
                        df["cleaned_abstract"].fillna("")).apply(clean)

abs_emb = embed_sliding(df["processed_text"].tolist())

# ── 5. Assign nearest reference label -----------------------------------------
sim         = cosine_similarity(abs_emb, ref_emb)
nearest_idx = np.argmax(sim, axis=1)
df["ref_id"] = [ref_labels[i] for i in nearest_idx]

# ── 6. Outlier removal (Isolation Forest) --------------------------------------
print(f"Original dataset size: {len(df)}")
scaled = StandardScaler().fit_transform(abs_emb)
outlier_pred = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1) \
               .fit_predict(scaled)
mask = outlier_pred == 1
df   = df[mask].reset_index(drop=True)
abs_emb = abs_emb[mask]
print(f"Filtered dataset size: {len(df)} (removed {(~mask).sum()} outliers)")

# ── 7. Macro-cluster remap -----------------------------------------------------
remap = {
    # 0 – Computer Vision & Image Processing
    **{str(i): "Computer Vision & Image Processing"
       for i in (2, 4, 5, 9, 15, 19, 20, 22, 23, 30, 35, 36, 45, 48)},

    # 1 – Natural Language Processing
    **{str(i): "Natural Language Processing"
       for i in (3, 24, 27, 34, 41, 49)},

    # 2 – Speech Recognition & Audio Processing
    **{str(i): "Speech Recognition & Audio Processing"
       for i in (7, 28, 33, 46)},

    # 3 – Neuromorphic Hardware Accelerators
    **{str(i): "Neuromorphic Hardware Accelerators"
       for i in (1, 38)},

    # 4 – Machine Learning Foundations
    **{str(i): "Machine Learning Foundations"
       for i in (0, 6, 8, 11, 12, 29, 31, 39, 40, 42)},

    # 5 – Biomarkers for Health & Performance
    **{str(i): "Biomarkers for Health & Performance"
       for i in (10, 13, 14, 16, 18, 26, 44, 47)},

    # 6 – Ethical & Creative AI
    **{str(i): "Ethical & Creative AI"
       for i in (17, 25, 43)},

    # 7 – Recommendation Systems
    **{str(i): "Recommendation Systems"
       for i in (21, 32, 37)}
}



# duplicate sanity‑check
if len(remap) != 50:
    missing = set(str(i) for i in range(50)) - set(remap)
    dupe_ids = [k for k in remap if list(remap).count(k) > 1]
    raise ValueError(f"Mapping invalid. Missing: {missing}, Duplicates: {dupe_ids}")

df["macro_cluster"] = [remap.get(lbl.split("_")[-1], "Unmapped")
                       for lbl in df["ref_id"]]

# ── 8. UMAP projection ---------------------------------------------------------
umap_xy = umap.UMAP(n_neighbors=15, min_dist=0.1,
                    metric='cosine', random_state=42).fit_transform(abs_emb)
df["x"], df["y"] = umap_xy[:, 0], umap_xy[:, 1]

# ── 9. Interactive Plotly UMAP -------------------------------------------------
palette = {
    "Computer Vision & Image Processing":        "#1f77b4",  # blue
    "Natural Language Processing":               "#ff7f0e",  # orange
    "Speech Recognition & Audio Processing":     "#2ca02c",  # green
    "Neuromorphic Hardware Accelerators":        "#9467bd",  # violet
    "Machine Learning Foundations":              "#d62728",  # red
    "Biomarkers for Health & Performance":       "#17becf",  # teal
    "Ethical & Creative AI":                     "#8c564b",  # brown
    "Recommendation Systems":                    "#bcbd22"   # olive
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
