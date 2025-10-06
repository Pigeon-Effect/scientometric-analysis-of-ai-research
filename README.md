# Mapping National Artificial Intelligence Research Authority  
### A Scientometric Analysis of AI Subdisciplines Using OpenAlex (2020–2025)  
**Author:** Julius Pfundstein  
**University of Leipzig — MSc Digital Humanities (2025)**  

---

## Overview
This repository contains the data processing, analysis, and visualization pipeline developed for my master’s thesis *“Mapping National Artificial Intelligence Research Authority”*.  
The project investigates how global AI research is distributed across subdisciplines and countries by applying large-scale bibliometric and scientometric methods to the **OpenAlex** knowledge graph.

---

## Research Goals
1. **Identify** the hierarchical subdisciplinary structure of AI research (2020–2025).  
2. **Quantify** national and regional research authority using citation-weighted measures.  
3. **Visualize** global specialization and dominance across AI macro-, meso-, and micro-fields.

---

## Data & Sources
- **Primary Source:** [OpenAlex API](https://openalex.org/)  
- **Corpus:** ~3.3 M AI-related works (2020–2025)  
- **Core Metadata:** title, abstract, cited_by_count, referenced_works, country_of_origin (fractional attribution)  
- **Storage:** SQLite database (`works_labeled.db`)  

---

## Methodology
- **Keyword Expansion:** 279 curated AI-related terms derived via survey-paper KeyBERT extraction.  
- **Hierarchical Topic Modeling:**  
  - H1 – 5 macro-domains  
  - H2 – 31 meso-domains  
  - H3 – 106 micro-clusters  
  - Vectorization: `SPECTER` embeddings + `TF–IDF` labeling  
- **Citation-weighted Authority Metric:**  
  \[
  \text{share}_{b}(h3)=\frac{\sum_{p\in h3}cited\_by\_count_p \cdot group\_fraction_{p,b}}{\sum_{p\in h3}cited\_by\_count_p}
  \]
- **Visualization:** UMAP layouts, citation-share heatmaps, percentile distributions, institutional leaderboards.  

---

## Repository Structure
