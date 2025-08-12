"""
h2_cluster.py  – Second‑hierarchy labelling for the natural‑science corpus
--------------------------------------------------------------------------
Macro‑cluster codes
    0 : Climate & Hydrology
    1 : Material Science
    2 : Energy & Thermodynamics
    3 : Agriculture & Forestry
    4 : Astrophysics & Quantumphysics
"""
import os, json, re, sqlite3, torch
import numpy as np, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ── File paths ────────────────────────────────────────────────────────────────
DB_PATH   = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\h1_cluster_subsets\natural_science_dataset.db"
JSON_PATH = r"C:\Users\Admin\Documents\Master-Thesis\results\topic_modeling\h2_cluster_natural_science\human_curation\cluster_abstracts_summary_002.json"

# ── 1.  Load curated reference documents  ─────────────────────────────────────
with open(JSON_PATH, encoding="utf-8") as f:
    curated = json.load(f)

ref_texts, ref_labels = [], []
for cid, entry in curated.items():
    lbl = f"cluster_{cid}" if cid.isdigit() else cid
    for section in ("most_central_abstracts", "random_abstracts"):
        for item in entry[section]:
            title = item.get("title") or ""
            abst  = item.get("cleaned_abstract") or ""
            ref_texts.append(f"{title.strip()}. {abst.strip()}")
            ref_labels.append(lbl)

# ── 2.  Embed reference set  ──────────────────────────────────────────────────
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

# ── 3.  Load full dataset  ────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql_query("""
        SELECT * FROM works_labeled
        WHERE cleaned_abstract IS NOT NULL
          AND language = 'en';
       """, conn)

# ── 4.  Light preprocessing & embedding  ──────────────────────────────────────
def clean(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    return " ".join(w for w in t.split() if len(w) > 2)

df["processed_text"] = (df["title"].fillna("") + ". " +
                        df["cleaned_abstract"].fillna("")).apply(clean)

abs_emb = embed_sliding(df["processed_text"].tolist())

# ── 5.  Nearest‑reference assignment  ─────────────────────────────────────────
sim          = cosine_similarity(abs_emb, ref_emb)
nearest_ref  = np.argmax(sim, axis=1)
df["ref_id"] = [ref_labels[i] for i in nearest_ref]

# ── 6.  Map original H1 cluster IDs → H2 codes (0‑4)  ─────────────────────────
H2_REMAP = {
    # 0 – Climate & Hydrology
    **{str(i): 0 for i in (1,5,11,15,18,43,49,0,44,31,13,27,32,12,16)},

    # 1 – Material Science
    **{str(i): 1 for i in (2,17,19,28,34,36,37,46,10,29,21,8,39,42)},

    # 2 – Energy & Thermodynamics
    **{str(i): 2 for i in (9,24,33,41,48,3,26,30)},

    # 3 – Agriculture & Forestry
    **{str(i): 3 for i in (4,6,7,20,22,25,35,47,38)},

    # 4 – Astrophysics & Quantumphysics
    **{str(i): 4 for i in (14,23,40,45)}
}

df["h2_cluster"] = [H2_REMAP.get(lbl.split("_")[-1], -1) for lbl in df["ref_id"]]

# ── 7.  Persist table with new column  ────────────────────────────────────────
df.to_sql("works_labeled", conn, if_exists="replace", index=False)
conn.close()

print("✓  Full corpus labelled with 'h2_cluster' (0‑4) and written back to database.")
