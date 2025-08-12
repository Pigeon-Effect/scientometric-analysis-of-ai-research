"""
h2_cluster_biomedical.py
────────────────────────────────────────────────────────────────────────────
Adds a second‑hierarchy label to every English record in the BIOMEDICAL
sample and produces a 2‑D UMAP visualisation.

Macro‑clusters (codes 0‑5)

    0 : Medical Imaging & Radiology
    1 : Neuroscience & Mental Health
    2 : Clinical Informatics & Public Health
    3 : Molecular Biology & Drug Discovery
    4 : Immunology & Infectious Disease
    5 : Genetics, Genomics & Oncology
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

# ── File paths ──────────────────────────────────────────────────────────────────
ABSTRACT_JSON_PATH = r"C:\Users\Admin\Documents\Master-Thesis\code\analysis\topic_modeling\sklearn_feature_extraction\h2_cluster_biomedical\resources\cluster_abstracts_summary.json"
DB_PATH   = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\h1_cluster_subsets\biomedical_sample.db"
OUTPUT_DIR  = r"C:\Users\Admin\Documents\Master-Thesis\code\analysis\topic_modeling\sklearn_feature_extraction\h2_cluster_biomedical\resources"
SVG_PATH = os.path.join(OUTPUT_DIR, "biomedical_umap.svg")

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

# ── 3. Load biomedical sample --------------------------------------------------
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

# ── 7. Macro‑cluster remap -----------------------------------------------------
remap = {
    # 0 – Medical Imaging & Radiology
    **{str(i): "Medical Imaging & Radiology"
       for i in (2, 3, 4, 8, 11, 18, 20, 22, 28, 43, 47)},

    # 1 – Neuroscience & Mental Health
    **{str(i): "Neuroscience & Mental Health"
       for i in (0, 6, 17, 23, 26, 27, 31, 33, 42, 48)},

    # 2 – Clinical Informatics & Public Health:
    **{str(i): "Clinical Informatics & Public Health"
       for i in (5, 12, 19, 38, 21, 25, 32, 35, 36, 39, 45, 49)},

    # 3 – Molecular Biology & Drug Discovery
    **{str(i): "Molecular Biology & Drug Discovery"
       for i in (9, 14, 15, 46)},

    # 4 – Immunology & Infectious Disease
    **{str(i): "Immunology & Infectious Disease"
       for i in (16, 29, 30, 40)},

    # 5 – Genetics & Genomics
    **{str(i): "Genetics & Genomics"
       for i in (1, 10, 24, 37, 41, 44, 7, 13, 34)},

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

# ── 9. SVG UMAP Visualization ------------------------------------------------
palette = {
    "Medical Imaging & Radiology":           "#3a86ff",  # Blue
    "Neuroscience & Mental Health":          "#ffbe0b",  # Yellow/Orange
    "Clinical Informatics & Public Health":  "#7209b7",  # Purple
    "Molecular Biology & Drug Discovery":    "#2dd4bf",  # Cyan/Teal
    "Immunology & Infectious Disease":       "#8b0000",  # Dark Red
    "Genetics & Genomics":                   "#fb5607",  # Red/Orange
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
