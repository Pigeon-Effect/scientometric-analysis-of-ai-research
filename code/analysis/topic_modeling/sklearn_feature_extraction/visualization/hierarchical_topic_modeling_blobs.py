#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render H3 cluster polygons from a Parquet that ALREADY CONTAINS polygons,
coloring each H3 polygon by its H1 cluster using a FIXED palette, applying a
force-directed separation to reduce overlaps, and labeling each cluster with a
3-digit code: h1h2h3 (one digit per level, numbers only).

Expected input columns:
  - h3_key
  - h3_key_polygon_json   (JSON: [[x,y], [x,y], ...] or null)
  - h1_cluster, h2_cluster, h3_cluster  (for color + labels)

Output:
  - single SVG with all cluster polygons, colored by H1, labeled, low overlap.
"""

import json
from typing import List, Tuple, Dict
from collections import Counter
import math
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.affinity import translate

# ──────────────────────────────────────────────────────────────────────────────
# Paths (edit if needed)
# ──────────────────────────────────────────────────────────────────────────────
INPUT_PATH  = r"C:\Users\Admin\Documents\Master-Thesis\results\topic_modeling\supercluster_umap_visulization\blob_umap_dataframe_relabelled.parquet"
OUTPUT_PATH = r"C:\Users\Admin\Documents\Master-Thesis\results\topic_modeling\hierarchical_topic_modeling_blobs_relabelled_005.svg"

# ──────────────────────────────────────────────────────────────────────────────
# Rendering params
# ──────────────────────────────────────────────────────────────────────────────
WIDTH = 1400
HEIGHT = 1000
PADDING = 24.0
FILL_OPACITY = 0.80   # increased to 60%
STROKE_WIDTH = 1.1

# Label styling
LABEL_FILL = "#111111"
LABEL_STROKE = "#ffffff"
LABEL_STROKE_WIDTH = 2.2
LABEL_FONT_FAMILY = "Times New Roman, serif"
LABEL_FONT_SIZE = 12

# ──────────────────────────────────────────────────────────────────────────────
# Fixed H1 → color mapping
# ──────────────────────────────────────────────────────────────────────────────
H1_COLOR_MAP: Dict[str, str] = {
    "0": "#7b2cbf",  # purple
    "1": "#3a86ff",  # blue
    "2": "#2dd4bf",  # teal
    "3": "#ffd500",  # yellow
    "4": "#fb5607",  # orange
}

# ──────────────────────────────────────────────────────────────────────────────
# Force-directed separation (circle proxies) — LESS AGGRESSIVE
# ──────────────────────────────────────────────────────────────────────────────
SEPARATION_ENABLED     = True

# Primary (circle-proxy) pass — softened
FD_MAX_ITERS           = 350
FD_STEP_SIZE           = 0.10   # was 0.15
FD_OVERLAP_GAIN        = 1.20   # was 1.6
FD_SPRING_K            = 0.05   # was 0.03 (slightly stronger tether)
FD_DAMPING             = 0.95   # was 0.96
FD_DISPLACEMENT_CAP    = 6.0    # was 8.0
FD_EPS                 = 1e-9

# Secondary refinement (true polygon overlaps) — softened
REFINE_ENABLED         = True
REFINE_ITERS           = 0     # was 120
REFINE_STEP            = 0.035  # was 0.05
REFINE_SPRING_K        = 0.03   # was 0.02 (slightly stronger tether)
REFINE_DAMPING         = 0.94   # was 0.95
REFINE_DISP_CAP        = 2.0    # was 2.5
REFINE_REPORT_EVERY    = 10

# ──────────────────────────────────────────────────────────────────────────────
# Geometry & SVG helpers
# ──────────────────────────────────────────────────────────────────────────────
def compute_bbox(polys: List[List[Tuple[float, float]]]) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for poly in polys:
        if not poly:
            continue
        px, py = zip(*poly)
        xs.extend(px); ys.extend(py)
    if not xs or not ys:
        raise ValueError("No valid polygon coordinates found.")
    return min(xs), min(ys), max(xs), max(ys)

def make_transform(xmin, ymin, xmax, ymax, width, height, pad):
    w = max(1e-12, xmax - xmin)
    h = max(1e-12, ymax - ymin)
    scale_x = (width - 2 * pad) / w
    scale_y = (height - 2 * pad) / h
    scale = min(scale_x, scale_y)  # preserve aspect

    def tf_point(p):
        x, y = p
        sx = pad + (x - xmin) * scale
        sy = height - pad - (y - ymin) * scale  # flip Y for SVG
        return sx, sy

    return tf_point, scale

def polygon_to_svg_path(poly: List[Tuple[float, float]], tf, precision: int = 2) -> str:
    coords = [tf(p) for p in poly]
    if not coords:
        return ""
    def fmt(v): return f"{v:.{precision}f}"
    d = f"M {fmt(coords[0][0])} {fmt(coords[0][1])} " + " ".join(
        f"L {fmt(x)} {fmt(y)}" for x, y in coords[1:]
    ) + " Z"
    return d

def render_svg(polys_h1_labels,
               width: int,
               height: int,
               padding: float,
               fill_opacity: float,
               stroke_width: float) -> str:
    """
    polys_h1_labels: list of (h3_key, polygon, h1_label, fill_hex, label_text)
    """
    polys_only = [poly for _, poly, _, _, _ in polys_h1_labels if poly]
    if not polys_only:
        raise ValueError("No polygons to render.")
    xmin, ymin, xmax, ymax = compute_bbox(polys_only)
    tf, _scale = make_transform(xmin, ymin, xmax, ymax, width, height, padding)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<style><![CDATA['
        f'  .poly {{ stroke:#333; stroke-opacity:0.9; fill-opacity:{fill_opacity}; stroke-width:{stroke_width}; }}',
        f'  .lbl-box {{ fill:#ffffff; stroke:#000000; stroke-width:1; rx:3; ry:3; }}',
        f'  .lbl {{ font-family:{LABEL_FONT_FAMILY}; font-size:{LABEL_FONT_SIZE}px; font-weight:600;'
        f'         fill:{LABEL_FILL}; }}'
        ']]></style>',
        '</defs>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />'
    ]

    # First pass: render all polygons
    for _, poly, _, fill_hex, label_text in polys_h1_labels:
        if not poly:
            continue
        path_d = polygon_to_svg_path(poly, tf=tf, precision=2)
        if path_d:
            parts.append(f'<path class="poly" d="{path_d}" fill="{fill_hex}" />')

    # Second pass: render all labels on top
    for _, poly, _, fill_hex, label_text in polys_h1_labels:
        if not poly:
            continue
        c = Polygon(poly).centroid
        sx, sy = tf((c.x, c.y))

        # Calculate text dimensions for background box
        text_width = len(label_text) * LABEL_FONT_SIZE * 0.6  # approximate width
        text_height = LABEL_FONT_SIZE * 1.2  # approximate height with some padding
        box_padding = 4

        # Add background box
        parts.append(
            f'<rect class="lbl-box" x="{sx - text_width/2 - box_padding:.2f}" y="{sy - text_height/2 - box_padding/2:.2f}" '
            f'width="{text_width + 2*box_padding:.2f}" height="{text_height + box_padding:.2f}" />'
        )

        # Add text label
        parts.append(
            f'<text class="lbl" x="{sx:.2f}" y="{sy:.2f}" text-anchor="middle" dominant-baseline="central">{label_text}</text>'
        )

    parts.append('</svg>')
    return "\n".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# Load polygons + labels from parquet
# ──────────────────────────────────────────────────────────────────────────────
def load_polygons_from_parquet(parquet_path: str):
    """
    Return (items, counts_drawn, counts_total)
      - items: list of (h3_key, polygon_pts, h1_label, fill_hex, label_text)
    """
    df = pd.read_parquet(parquet_path)

    expected_cols = {"h3_key", "h3_key_polygon_json", "h1_cluster", "h2_cluster", "h3_cluster"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Input parquet must contain {expected_cols}. Found: {sorted(df.columns)}")

    # map h3_key -> (h1,h2,h3)
    triple_map = df.groupby("h3_key")[["h1_cluster", "h2_cluster", "h3_cluster"]].first()

    # counters (sanity)
    counts_total = triple_map["h1_cluster"].astype(str).value_counts()
    counts_total = Counter({str(k): int(v) for k, v in counts_total.items()})

    by_key: Dict[str, Tuple[List[Tuple[float, float]], str, str]] = {}
    for key, poly_js in zip(df["h3_key"].values, df["h3_key_polygon_json"].values):
        if poly_js is None or (isinstance(poly_js, float) and np.isnan(poly_js)):
            continue
        try:
            pts = poly_js if isinstance(poly_js, (list, tuple)) else json.loads(poly_js)
            poly: List[Tuple[float, float]] = []
            for p in pts:
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    x, y = float(p[0]), float(p[1])
                    poly.append((x, y))
            if len(poly) >= 3:
                k = str(key)
                h1 = int(triple_map.loc[k, "h1_cluster"])
                h2 = int(triple_map.loc[k, "h2_cluster"])
                h3 = int(triple_map.loc[k, "h3_cluster"])
                # label: one digit per level (numbers only)
                label_text = f"{h1%10}{h2%10}{h3%10}"
                if (k not in by_key) or (len(poly) > len(by_key[k][0])):
                    by_key[k] = (poly, str(h1), label_text)
        except Exception:
            continue

    items = []
    for k in sorted(by_key.keys(), key=lambda s: (len(s), s)):
        poly, h1, label_text = by_key[k]
        fill = H1_COLOR_MAP.get(h1, "#cccccc")
        items.append((k, poly, h1, fill, label_text))

    counts_drawn = Counter([h1 for (_, _, h1, _, _) in items])
    return items, counts_drawn, counts_total

# ──────────────────────────────────────────────────────────────────────────────
# Separation core — circle proxies, then polygon refine
# ──────────────────────────────────────────────────────────────────────────────
def polygon_centroid_area(coords: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
    poly = Polygon(coords)
    c = np.array([poly.centroid.x, poly.centroid.y], dtype=float)
    return c, float(poly.area)

def circles_overlap(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> Tuple[bool, float, np.ndarray]:
    """
    Returns (overlaps, penetration_depth, direction_from_1_to_2_unit)
    If centers coincide, returns a fixed unit direction.
    """
    dvec = c2 - c1
    dist = float(np.hypot(dvec[0], dvec[1]))
    rsum = r1 + r2
    if dist < rsum - FD_EPS:
        if dist < FD_EPS:
            u = np.array([1.0, 0.0], dtype=float)
        else:
            u = dvec / max(dist, FD_EPS)
        return True, (rsum - dist), u
    return False, 0.0, np.zeros(2, dtype=float)

def run_force_directed_separation(items):
    """
    items: list of (h3_key, coords, h1, fill, label)
    returns same structure with translated coords
    """
    n = len(items)
    if n <= 1 or not SEPARATION_ENABLED:
        return items

    # prepare state
    centroids = np.zeros((n, 2), dtype=float)
    radii     = np.zeros((n,), dtype=float)
    deltas    = np.zeros((n, 2), dtype=float)
    coords_list: List[List[Tuple[float, float]]] = []
    keys, h1s, fills, labels = [], [], [], []

    for i, (k, coords, h1, fill, lab) in enumerate(items):
        c, area = polygon_centroid_area(coords)
        r = math.sqrt(max(area, FD_EPS) / math.pi)
        centroids[i] = c
        radii[i] = r
        coords_list.append(coords)
        keys.append(k)
        h1s.append(h1)
        fills.append(fill)
        labels.append(lab)

    # pass 1: circle proxies (less aggressive)
    for it in range(FD_MAX_ITERS):
        forces = np.zeros_like(deltas)
        overlaps = 0

        for i in range(n):
            ci = centroids[i] + deltas[i]
            ri = radii[i]
            for j in range(i + 1, n):
                cj = centroids[j] + deltas[j]
                rj = radii[j]
                ov, pen, u = circles_overlap(ci, ri, cj, rj)
                if not ov:
                    continue
                overlaps += 1
                kick = FD_OVERLAP_GAIN * pen * u
                forces[i] -= 0.5 * kick
                forces[j] += 0.5 * kick

        # spring tether
        forces += -FD_SPRING_K * deltas

        # integrate
        step = FD_STEP_SIZE * forces
        deltas = (deltas + step) * FD_DAMPING

        # cap displacement
        norms = np.linalg.norm(deltas, axis=1)
        too_far = norms > FD_DISPLACEMENT_CAP
        if np.any(too_far):
            deltas[too_far] *= (FD_DISPLACEMENT_CAP / norms[too_far])[:, None]

        if it % 10 == 0 or overlaps == 0 or it == FD_MAX_ITERS - 1:
            md = float(np.max(np.linalg.norm(deltas, axis=1)))
            print(f"[separate/circle] iter {it:03d}: overlaps={overlaps}, max|Δ|={md:.3f}")
        if overlaps == 0:
            break

    # apply to polygons
    moved_polys = []
    for i in range(n):
        dx, dy = deltas[i]
        moved = [(x + dx, y + dy) for (x, y) in coords_list[i]]
        moved_polys.append(moved)

    # optional refine pass using real polygon intersections (less aggressive)
    if REFINE_ENABLED:
        polys = [Polygon(p) for p in moved_polys]
        deltas2 = np.zeros_like(deltas)

        for it in range(REFINE_ITERS):
            overlaps = 0
            forces = np.zeros_like(deltas2)

            for i in range(n):
                pi = polys[i]
                ci = np.array([pi.centroid.x, pi.centroid.y], dtype=float)
                for j in range(i + 1, n):
                    pj = polys[j]
                    if not pi.intersects(pj):
                        continue
                    inter = pi.intersection(pj)
                    if inter.is_empty:
                        continue
                    overlaps += 1

                    cj = np.array([pj.centroid.x, pj.centroid.y], dtype=float)
                    dvec = cj - ci
                    dist = float(np.hypot(dvec[0], dvec[1]))
                    u = dvec / max(dist, 1e-9)

                    kick = u  # unit push
                    forces[i] -= 0.5 * kick
                    forces[j] += 0.5 * kick

            # spring tether
            forces += -REFINE_SPRING_K * deltas2

            # integrate
            step = REFINE_STEP * forces
            deltas2 = (deltas2 + step) * REFINE_DAMPING

            # cap refine displacement
            norms = np.linalg.norm(deltas2, axis=1)
            too_far = norms > REFINE_DISP_CAP
            if np.any(too_far):
                deltas2[too_far] *= (REFINE_DISP_CAP / norms[too_far])[:, None]

            # apply incremental translation
            for i in range(n):
                dx, dy = step[i]
                if abs(dx) + abs(dy) > 0:
                    polys[i] = translate(polys[i], xoff=dx, yoff=dy)

            if (it % REFINE_REPORT_EVERY == 0) or overlaps == 0 or it == REFINE_ITERS - 1:
                md = float(np.max(np.linalg.norm(deltas2, axis=1)))
                print(f"[refine/polygon] iter {it:03d}: overlaps={overlaps}, max|Δ_ref|={md:.3f}")
            if overlaps == 0:
                break

        moved_polys = [list(poly.exterior.coords)[:-1] for poly in polys]

    out = []
    for i in range(n):
        out.append((keys[i], moved_polys[i], h1s[i], fills[i], labels[i]))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    items, counts_drawn, counts_total = load_polygons_from_parquet(INPUT_PATH)
    if not items:
        raise SystemExit("No valid polygons found in parquet (h3_key_polygon_json is empty).")

    missing_h1 = sorted(set(counts_total.keys()) - set(counts_drawn.keys()))
    if missing_h1:
        print(f"[!] H1 clusters present in data but with NO drawn polygons: {missing_h1}")

    if SEPARATION_ENABLED:
        print("[i] Running force-directed separation… (less aggressive)")
        items = run_force_directed_separation(items)
        print("[i] Separation complete.")

    svg = render_svg(
        polys_h1_labels=items,
        width=WIDTH,
        height=HEIGHT,
        padding=PADDING,
        fill_opacity=FILL_OPACITY,
        stroke_width=STROKE_WIDTH
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"[✓] Wrote SVG with {len(items)} polygons → {OUTPUT_PATH}")
    print(f"[i] H1 palette: {H1_COLOR_MAP}")
    print(f"[i] H1 counts (total H3 groups): {dict(counts_total)}")
    print(f"[i] H1 counts (drawn polygons): {dict(counts_drawn)}")

if __name__ == "__main__":
    main()
