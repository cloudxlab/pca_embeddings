# Domain-Focused PCA on Text Embeddings Improves Semantic Retrieval

> **Paper:** *Domain-Focused PCA on Text Embeddings Improves Semantic Retrieval: A Medical Domain Study*
> [`paper.pdf`](paper.pdf) · [`paper.html`](paper.html)

General-purpose text embeddings (e.g. OpenAI `text-embedding-3-small`) are trained on broad web data, encoding variation across all knowledge domains. When used for retrieval in a narrow domain, most dimensions carry cross-domain noise. This project shows that fitting PCA on a domain-specific document corpus and projecting embeddings into the top-k principal components consistently improves retrieval quality — with no fine-tuning required.

**Best result:** PCA-32 (corpus-only fit) on a 20-topic medical corpus achieves **MAP 0.9203 vs 0.8750 baseline (+5.2%)**, similarity gap 2.5×, and 48× storage compression.

---

## Key Findings

| Hypothesis | Result |
|---|---|
| H1: Optimal PCA dim ≈ 1.6× number of topics | Confirmed (32 dims for 20 topics) |
| H2: PCA vs Random Projection | PCA wins all 8 dims — domain axes are essential |
| H3: Corpus-only fit outperforms query+corpus fit | Confirmed — best overall result |
| H4: PCA more robust to hard negatives | Not confirmed — baseline slightly more robust |
| H5: PCA gain grows (not shrinks) with corpus diversity | Confirmed — largest gain at 20 topics |

---

## Quickstart

```bash
pip install openai scikit-learn numpy pandas matplotlib seaborn
export OPENAI_API_KEY=sk-...

# Run the full experiment (v2: 20 topics, 5 hypotheses)
python3 pca_experiment_v2.py

# Or the original smaller experiment (10 topics)
python3 pca_experiment.py

# Regenerate figures
python3 generate_figures.py
```

Embeddings are cached in `embeddings_cache.json` after the first run — subsequent runs are free.

---

## Repository Structure

```
pca_embedding/
├── pca_experiment.py          # v1: 10 topics × 6 docs, baseline experiment
├── pca_experiment_v2.py       # v2: 20 topics × 15 docs, 5 hypotheses
├── generate_figures.py        # Generates all 5 paper figures
│
├── figures/
│   ├── fig1_map_vs_dims.png   # MAP vs PCA dimensionality
│   ├── fig2_pca_vs_rp.png     # PCA vs Random Projection
│   ├── fig3_simgap.png        # Similarity gap across dims
│   ├── fig4_topic_diversity.png  # PCA gain vs # topics
│   └── fig5_variance_map.png  # Variance explained vs MAP
│
├── pca_results.csv            # v1 raw results
├── pca_results_v2.csv         # v2 raw results
├── embeddings_cache.json      # Cached OpenAI embeddings
│
├── paper.md                   # Full paper (Markdown source)
├── references.bib             # BibTeX references
├── paper.pdf                  # Compiled PDF
├── paper.html                 # Standalone HTML (with figures embedded)
└── paper.docx                 # Word document
```

---

## How It Works

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import numpy as np

# 1. Embed your domain corpus
corpus_embeddings  # shape (N, 1536)
query_embeddings   # shape (Q, 1536)

# 2. Fit PCA on corpus only (not queries)
pca = PCA(n_components=32, random_state=42)
pca.fit(corpus_embeddings)

# 3. Project at index time and query time
corpus_reduced = pca.transform(corpus_embeddings)   # (N, 32)
query_reduced  = pca.transform(query_embeddings)    # (Q, 32)

# 4. Retrieve with cosine similarity
corpus_norm = normalize(corpus_reduced)
query_norm  = normalize(query_reduced)
scores = query_norm @ corpus_norm.T                 # (Q, N)
```

---

## Corpus

| | v1 | v2 |
|---|---|---|
| Topics | 10 | 20 |
| Docs / topic | 6 | 15 |
| Noise docs | 8 | 20 |
| Hard negatives | — | 10 |
| Queries | 10 | 20 |
| Total embedded | 78 | 340 |

**Clinical topics covered:** Type 2 Diabetes, Myocardial Infarction, Antibiotic Resistance, Pulmonary Embolism, ACE Inhibitors, Stroke, COVID-19 Vaccines, Chronic Kidney Disease, Asthma, Major Depressive Disorder, Parkinson's Disease, Alzheimer's Disease, Rheumatoid Arthritis, Sepsis, Liver Cirrhosis, Tuberculosis, Breast Cancer, Schizophrenia, Inflammatory Bowel Disease, Thyroid Disorders.

---

## Results Summary (v2, 20 topics)

| Dims | Method | MAP | NDCG@10 | SimGap | Compression |
|-----:|--------|----:|--------:|-------:|------------:|
| 1536 | Baseline | 0.8750 | 0.9235 | 0.250 | 1× |
| **32** | **PCA corpus-only** | **0.9203** | **0.9498** | **0.615** | **48×** |
| 48 | PCA corpus-only | 0.9142 | 0.9525 | 0.554 | 32× |
| 64 | PCA corpus-only | 0.9137 | 0.9465 | 0.512 | 24× |
| 32 | Random Projection | 0.3582 | 0.4356 | 0.237 | 48× |

---

## Requirements

- Python 3.10+
- `openai` — embedding API
- `scikit-learn` — PCA
- `numpy`, `pandas` — data handling
- `matplotlib`, `seaborn` — figures

```bash
pip install openai scikit-learn numpy pandas matplotlib seaborn
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{giri2026pca,
  title   = {Domain-Focused PCA on Text Embeddings Improves Semantic Retrieval:
             A Medical Domain Study},
  author  = {Giri, Sandeep},
  year    = {2026},
  url     = {https://github.com/sandeepgiri/pca_embedding}
}
```

---

## License

MIT
