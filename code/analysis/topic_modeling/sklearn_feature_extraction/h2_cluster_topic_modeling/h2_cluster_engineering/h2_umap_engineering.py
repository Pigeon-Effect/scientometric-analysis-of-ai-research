"""
h2_umap_engineering.py
────────────────────────────────────────────────────────────────────────────
Creates a UMAP to countercheck the clustermapping created earlier with the
interactive umap and the extracted abstracts based on the
ENGINEERING / APPLIED‑SCIENCES sample.

Macro‑clusters (codes 0‑5)

    0 : Cybersecurity & Privacy
    1 : IoT & Edge Computing
    2 : Anomaly & Fault Detection
    3 : Energy Technologies
    4 : Autonomous Mobility
    5 : Robotics & Mechatronics
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
import plotly.express as px

# ── File & folder locations ───────────────────────────────────────────────────
DB_PATH          = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\h1_cluster_subsets\engineering_sample.db"
OUTPUT_DIR       = r"C:\Users\Admin\Documents\Master-Thesis\code\analysis\topic_modeling\sklearn_feature_extraction\h2_cluster_engineering\resources"
UMAP_HTML_PATH   = os.path.join(OUTPUT_DIR, "interactive_umap.html")
ABSTRACT_JSON_PATH = os.path.join(OUTPUT_DIR, "cluster_abstracts_summary.json")  # curated refs

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

# ── 7. Macro‑cluster remap -----------------------------------------------------
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
    "Cybersecurity & Privacy":      "#3a86ff",
    "IoT & Edge Computing":         "#ffbe0b",
    "Anomaly & Fault Detection":    "#7209b7",
    "Energy Technologies":          "#fb5607",
    "Autonomous Mobility":          "#2dd4bf",
    "Robotics & Mechatronics":      "#8b0000"
}

df_plot = df.copy()
order = df_plot["macro_cluster"].value_counts().index.tolist()
fig = px.scatter(df_plot, x="x", y="y",
                 color="macro_cluster",
                 category_orders={"macro_cluster": order},
                 color_discrete_map=palette,
                 hover_data={"id": True},
                 height=900, width=1200,
                 title="Interactive UMAP • Engineering & Applied Sciences")

fig.update_traces(marker=dict(size=4, opacity=0.7))
fig.write_html(UMAP_HTML_PATH)
print("✓ Interactive UMAP saved to:", UMAP_HTML_PATH)
