#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a tidy DataFrame for hierarchical blob-UMAP visualization (H3 polygons only),
using density-based core-shape extraction, then SCALE polygon areas ∝ number of works,
and SAVE the polygons.

Reads:
    - blob_sample_embeddings.npy
    - blob_sample_metadata.parquet  (must include: h1_cluster, h2_cluster, h3_cluster, h3_key)

Does:
    - UMAP (fit on subsample, transform all)
    - KDE density-contour per H3 (mass threshold, anti-clipping grid padding)
    - Area scaling so Area_i ≈ AREA_PER_PAPER * N_i
    - Save parquet WITH polygons (h3_key_polygon_json)
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import umap.umap_ as umap
from shapely.geometry import Polygon
from shapely.affinity import scale as shp_scale
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
UMAP_N_NEIGH = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

# Density-based polygon extraction
DENSITY_MASS_THRESHOLD = 0.50   # fraction of density mass to enclose in polygon
GRID_RES = 200                  # KDE grid resolution per axis
GRID_PADDING_SIGMA = 2.0        # initial padding in KDE-effective std units
MAX_PADDING_TRIES   = 3         # increase padding up to this many attempts
FRAME_TOUCH_TOL     = 0.08      # fraction of padding → consider polygon touching frame

# Area scaling (Area target = AREA_PER_PAPER * N_cluster)
AREA_PER_PAPER   = 0.0003        # UMAP^2 per paper (tune!)
SCALE_CLAMP_MIN  = 0.50         # min multiplicative scale (to avoid vanishing)
SCALE_CLAMP_MAX  = 2.50         # max multiplicative scale (to avoid explosions)

# I/O
DEFAULT_OUTPUT_DIR = r"C:\Users\Admin\Documents\Master-Thesis\results\topic_modeling\supercluster_umap_visulization"
EMB_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "blob_sample_embeddings.npy")
META_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "blob_sample_metadata_relabelled.parquet")
OUT_PATH  = os.path.join(DEFAULT_OUTPUT_DIR, "blob_umap_dataframe_relabelled.parquet")

# UMAP fit size (subsample for speed)
UMAP_FIT_N = 5_000


# ──────────────────────────────────────────────────────────────────────────────
# KDE density polygon with anti-clipping padding
# ──────────────────────────────────────────────────────────────────────────────
def compute_density_polygon(points: np.ndarray,
                            mass_threshold: float = DENSITY_MASS_THRESHOLD,
                            grid_res: int = GRID_RES,
                            pad_sigmas: float = GRID_PADDING_SIGMA,
                            max_retries: int = MAX_PADDING_TRIES,
                            frame_touch_tol: float = FRAME_TOUCH_TOL):
    """
    KDE + mass contour polygon. Pads the grid beyond data bounds by
    (pad_sigmas * effective_std) and retries if the polygon touches the frame.
    Returns list of (x, y) or None.
    """
    if points.shape[0] < 5:
        return None

    kde = gaussian_kde(points.T)

    xmin_data, ymin_data = points.min(axis=0)
    xmax_data, ymax_data = points.max(axis=0)

    if not np.isfinite([xmin_data, ymin_data, xmax_data, ymax_data]).all() or \
       (xmax_data == xmin_data) or (ymax_data == ymin_data):
        return None

    # Effective std from smoothed covariance
    cov = kde.covariance
    eff_std_x = float(np.sqrt(cov[0, 0]))
    eff_std_y = float(np.sqrt(cov[1, 1]))

    for attempt in range(max_retries):
        pad_x = (pad_sigmas * (attempt + 1)) * max(eff_std_x, 1e-9)
        pad_y = (pad_sigmas * (attempt + 1)) * max(eff_std_y, 1e-9)

        x_min = xmin_data - pad_x
        x_max = xmax_data + pad_x
        y_min = ymin_data - pad_y
        y_max = ymax_data + pad_y

        X, Y = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res)
        )
        grid = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(grid).reshape(grid_res, grid_res)

        flat = Z.ravel()
        order = np.argsort(flat)[::-1]
        sorted_vals = flat[order]
        cumsum = np.cumsum(sorted_vals)
        cumsum /= cumsum[-1]
        idx = np.searchsorted(cumsum, mass_threshold)
        if idx >= sorted_vals.size:
            continue
        level = sorted_vals[idx]

        fig, ax = plt.subplots()
        try:
            CS = ax.contour(X, Y, Z, levels=[level])
        finally:
            plt.close(fig)
        if not CS.allsegs or not CS.allsegs[0]:
            continue

        max_seg = max(CS.allsegs[0], key=lambda arr: Polygon(arr).area)
        poly = Polygon(max_seg)
        if (not poly.is_valid) or poly.is_empty:
            continue

        # Check frame touch
        minx, miny, maxx, maxy = poly.bounds
        touch_xmin = (minx - x_min) < (frame_touch_tol * pad_x)
        touch_xmax = (x_max - maxx) < (frame_touch_tol * pad_x)
        touch_ymin = (miny - y_min) < (frame_touch_tol * pad_y)
        touch_ymax = (y_max - maxy) < (frame_touch_tol * pad_y)
        touches = touch_xmin or touch_xmax or touch_ymin or touch_ymax

        if touches and (attempt + 1) < max_retries:
            continue

        coords = list(poly.exterior.coords)[:-1]
        return [(float(x), float(y)) for (x, y) in coords]

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Polygon area scaling to match target area ∝ N
# ──────────────────────────────────────────────────────────────────────────────
def scale_polygon_to_target_area(coords, target_area: float):
    """
    Uniformly scale polygon about its centroid so area ≈ target_area.
    Returns list[(x, y)] or original coords if scaling not possible.
    """
    if not coords or target_area <= 0:
        return coords

    poly = Polygon(coords)
    if (not poly.is_valid) or poly.is_empty:
        return coords

    current_area = poly.area
    if current_area <= 0:
        return coords

    s = np.sqrt(target_area / current_area)
    # Clamp to avoid extreme distortions
    s = float(np.clip(s, SCALE_CLAMP_MIN, SCALE_CLAMP_MAX))

    c = poly.centroid
    poly_scaled = shp_scale(poly, xfact=s, yfact=s, origin=(c.x, c.y))
    if (not poly_scaled.is_valid) or poly_scaled.is_empty:
        return coords

    out_coords = list(poly_scaled.exterior.coords)[:-1]
    return [(float(x), float(y)) for (x, y) in out_coords]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("UMAP VISUALIZATION DATAFRAME CREATION (Density polygons, area∝count, SAVED)")
    print("=" * 60)

    # 1) Load precomputed data
    print("[1/4] Loading embeddings and metadata...")
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(f"Embeddings not found: {EMB_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata not found: {META_PATH}")

    emb = np.load(EMB_PATH, mmap_mode="r")
    emb32 = np.asarray(emb, dtype=np.float32)
    meta = pd.read_parquet(META_PATH)

    if emb.shape[0] != len(meta):
        raise ValueError(f"Row mismatch: embeddings={emb.shape[0]} vs metadata={len(meta)}")
    print(f"[i] Loaded {emb.shape[0]:,} rows, embedding dim={emb.shape[1]}")

    required = {"h1_cluster", "h2_cluster", "h3_cluster", "h3_key"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    # 2) UMAP fit+transform
    print("[2/4] Fitting and transforming UMAP...")
    rng = np.random.default_rng(SEED)
    fit_n = min(UMAP_FIT_N, emb32.shape[0])
    fit_idx = rng.choice(emb32.shape[0], size=fit_n, replace=False)

    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGH,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=SEED,
        verbose=True
    )
    reducer.fit(emb32[fit_idx])
    umap_xy = reducer.transform(emb32)
    meta["umap_x"] = umap_xy[:, 0]
    meta["umap_y"] = umap_xy[:, 1]

    # 3) KDE polygons per H3 key, then area scaling
    print("[3/4] Computing density-based polygons + area scaling...")
    # Cluster counts from the metadata parquet itself (works per H3 in the sample)
    counts = meta.groupby("h3_key", observed=True).size().astype(int).to_dict()

    polygons = {}
    success = 0
    for key in tqdm(meta["h3_key"].unique(), desc="H3 KDE polygons", unit="cluster"):
        pts = meta.loc[meta["h3_key"] == key, ["umap_x", "umap_y"]].to_numpy()
        poly_coords = compute_density_polygon(
            pts,
            mass_threshold=DENSITY_MASS_THRESHOLD,
            grid_res=GRID_RES,
            pad_sigmas=GRID_PADDING_SIGMA,
            max_retries=MAX_PADDING_TRIES,
            frame_touch_tol=FRAME_TOUCH_TOL
        )
        if poly_coords is None:
            polygons[key] = None
            continue

        # Target area proportional to number of works in this (sampled) cluster
        N = counts.get(key, 0)
        target_area = AREA_PER_PAPER * float(N) if N > 0 else 0.0

        scaled_coords = scale_polygon_to_target_area(poly_coords, target_area)
        if scaled_coords:
            polygons[key] = json.dumps(scaled_coords)
            success += 1
        else:
            polygons[key] = json.dumps(poly_coords)  # fallback to unscaled

    meta["h3_key_polygon_json"] = meta["h3_key"].map(polygons)
    print(f"[i] Polygons created & scaled: {success}/{len(polygons)}")

    # 4) Save final parquet WITH polygons
    print("[4/4] Saving final parquet (with polygons)...")
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    out_df = meta[[
        "paper_id",
        "title",
        "abstract",
        "h1_cluster",
        "h2_cluster",
        "h3_cluster",
        "h3_key",
        "umap_x",
        "umap_y",
        "h3_key_polygon_json",
    ]]
    out_df.to_parquet(OUT_PATH, index=False)

    print("=" * 60)
    print(f"[✓] Saved to {OUT_PATH}")
    print(f"[✓] Polygons (scaled): {success}/{len(polygons)}")
    print(f"[i] AREA_PER_PAPER={AREA_PER_PAPER}, "
          f"SCALE_CLAMP=[{SCALE_CLAMP_MIN}, {SCALE_CLAMP_MAX}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
