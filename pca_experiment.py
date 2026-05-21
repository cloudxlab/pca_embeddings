#!/usr/bin/env python3
"""
PCA Embedding Experiment — Medical Domain
Tests whether PCA dimensionality reduction improves or degrades
semantic matching (retrieval) performance on domain-specific medical text.

Metrics:
  - Mean Average Precision (MAP)
  - NDCG@10
  - Mean similarity gap (relevant vs irrelevant pairs)

Run: python3 pca_experiment.py
     Requires: OPENAI_API_KEY env var
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

CACHE_FILE = Path("embeddings_cache.json")
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims
PCA_DIMS = [32, 64, 128, 256, 512, 768, 1024]

# ---------------------------------------------------------------------------
# Medical domain corpus
# Each entry: (text, topic_id)
# Queries are paired with topic_ids to define relevance
# ---------------------------------------------------------------------------

CORPUS_LABELED = [
    # --- Diabetes & Metabolic (topic 0) ---
    ("Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus, reducing hepatic glucose production.", 0),
    ("Glycemic control in type 2 diabetes is achieved through lifestyle modification, metformin, and GLP-1 receptor agonists.", 0),
    ("SGLT2 inhibitors such as empagliflozin lower blood glucose by preventing renal glucose reabsorption in diabetic patients.", 0),
    ("HbA1c monitoring every 3 months guides adjustment of antidiabetic therapy in type 2 diabetes management.", 0),
    ("Insulin resistance in type 2 diabetes is closely associated with obesity, sedentary lifestyle, and visceral adiposity.", 0),
    ("Bariatric surgery leads to significant remission of type 2 diabetes through metabolic and hormonal mechanisms.", 0),

    # --- Myocardial Infarction & Cardiac (topic 1) ---
    ("Acute myocardial infarction presents with crushing substernal chest pain radiating to the left arm and diaphoresis.", 1),
    ("Elevated troponin I and troponin T are the gold standard biomarkers for diagnosing myocardial infarction.", 1),
    ("Primary percutaneous coronary intervention is the preferred reperfusion strategy for ST-elevation myocardial infarction.", 1),
    ("Aspirin and dual antiplatelet therapy with clopidogrel are standard post-MI treatment to prevent re-thrombosis.", 1),
    ("Left ventricular ejection fraction assessment by echocardiography is critical after myocardial infarction.", 1),
    ("Beta-blockers reduce mortality after myocardial infarction by decreasing myocardial oxygen demand.", 1),

    # --- Antibiotic Resistance (topic 2) ---
    ("Methicillin-resistant Staphylococcus aureus (MRSA) is resistant to beta-lactam antibiotics via the mecA gene.", 2),
    ("Vancomycin remains the drug of choice for serious MRSA infections, though vancomycin-resistant strains have emerged.", 2),
    ("Horizontal gene transfer via plasmids is the primary mechanism of antibiotic resistance spread among bacteria.", 2),
    ("Carbapenem-resistant Enterobacteriaceae (CRE) pose a critical threat due to limited therapeutic options.", 2),
    ("Antibiotic stewardship programs reduce inappropriate antibiotic prescribing and slow resistance development.", 2),
    ("Biofilm formation in Pseudomonas aeruginosa contributes significantly to antibiotic tolerance in chronic infections.", 2),

    # --- Pulmonary Embolism (topic 3) ---
    ("Pulmonary embolism classically presents with sudden onset dyspnea, pleuritic chest pain, and hypoxia.", 3),
    ("The Wells score and D-dimer test stratify clinical probability of pulmonary embolism in emergency settings.", 3),
    ("CT pulmonary angiography is the definitive imaging modality for confirming pulmonary embolism diagnosis.", 3),
    ("Anticoagulation with heparin or direct oral anticoagulants is the cornerstone of pulmonary embolism treatment.", 3),
    ("Massive pulmonary embolism with hemodynamic compromise may require systemic thrombolysis or embolectomy.", 3),
    ("Inferior vena cava filters are used when anticoagulation is contraindicated in pulmonary embolism patients.", 3),

    # --- ACE Inhibitors & Hypertension (topic 4) ---
    ("ACE inhibitors block the conversion of angiotensin I to angiotensin II, reducing vasoconstriction and aldosterone.", 4),
    ("Lisinopril and enalapril are commonly used ACE inhibitors for hypertension, heart failure, and diabetic nephropathy.", 4),
    ("Dry cough is a common side effect of ACE inhibitors caused by bradykinin accumulation in the airways.", 4),
    ("Angiotensin receptor blockers (ARBs) are used as ACE inhibitor alternatives in patients with intolerable cough.", 4),
    ("ACE inhibitors reduce proteinuria and slow progression of chronic kidney disease in diabetic patients.", 4),
    ("Hyperkalemia and acute kidney injury are serious adverse effects of ACE inhibitors requiring monitoring.", 4),

    # --- Stroke (topic 5) ---
    ("Hypertension, atrial fibrillation, diabetes, and smoking are major modifiable risk factors for ischemic stroke.", 5),
    ("Ischemic stroke is caused by thrombotic or embolic occlusion of cerebral arteries, causing focal neurological deficits.", 5),
    ("IV alteplase (tPA) within 4.5 hours of symptom onset is the standard thrombolytic therapy for ischemic stroke.", 5),
    ("Mechanical thrombectomy has revolutionized treatment of large vessel occlusion strokes up to 24 hours after onset.", 5),
    ("Hemorrhagic stroke accounts for 15% of strokes and carries higher short-term mortality than ischemic stroke.", 5),
    ("Secondary stroke prevention includes antiplatelet agents, statins, blood pressure control, and anticoagulation for AF.", 5),

    # --- COVID-19 & Vaccines (topic 6) ---
    ("mRNA vaccines against COVID-19 encode the spike protein and elicit strong humoral and cellular immune responses.", 6),
    ("BNT162b2 (Pfizer-BioNTech) showed 95% efficacy against symptomatic COVID-19 in phase 3 clinical trials.", 6),
    ("COVID-19 vaccine effectiveness wanes over time, supporting the use of booster doses for continued protection.", 6),
    ("SARS-CoV-2 variants such as Omicron partially evade vaccine-induced immunity, reducing neutralizing antibody titers.", 6),
    ("COVID-19 vaccines significantly reduce risk of severe disease, hospitalization, and death across age groups.", 6),
    ("Myocarditis is a rare adverse event associated with mRNA COVID-19 vaccines, particularly in young males.", 6),

    # --- Chronic Kidney Disease (topic 7) ---
    ("CKD staging by GFR and albuminuria guides treatment intensity and dialysis planning in kidney disease patients.", 7),
    ("RAAS blockade with ACE inhibitors or ARBs is the primary renoprotective strategy in CKD with proteinuria.", 7),
    ("Anemia of CKD results from reduced erythropoietin production and is treated with ESA agents and iron therapy.", 7),
    ("Secondary hyperparathyroidism in CKD requires phosphate binders, vitamin D analogs, and calcimimetics.", 7),
    ("Hemodialysis three times weekly is the most common renal replacement therapy for end-stage kidney disease.", 7),
    ("Peritoneal dialysis offers home-based renal replacement and preserves residual kidney function longer than HD.", 7),

    # --- Asthma (topic 8) ---
    ("Asthma is characterized by reversible airflow obstruction, airway hyperresponsiveness, and eosinophilic inflammation.", 8),
    ("Inhaled corticosteroids are the cornerstone of asthma control, reducing airway inflammation and exacerbation frequency.", 8),
    ("Short-acting beta-2 agonists like salbutamol provide rapid bronchodilation for acute asthma symptom relief.", 8),
    ("Leukotriene receptor antagonists such as montelukast are adjunctive therapy for mild-to-moderate persistent asthma.", 8),
    ("Biologic agents targeting IL-5, IgE, or TSLP are used for severe eosinophilic or allergic asthma.", 8),
    ("Peak flow monitoring and written asthma action plans reduce asthma-related hospitalizations and mortality.", 8),

    # --- Depression (topic 9) ---
    ("Major depressive disorder is treated with SSRIs as first-line pharmacotherapy due to efficacy and tolerability.", 9),
    ("Cognitive behavioral therapy (CBT) is as effective as antidepressants for mild-to-moderate major depression.", 9),
    ("Treatment-resistant depression responds to augmentation with lithium, atypical antipsychotics, or ketamine.", 9),
    ("SNRIs like venlafaxine and duloxetine are effective for comorbid depression and anxiety or neuropathic pain.", 9),
    ("Electroconvulsive therapy (ECT) remains the most effective treatment for severe refractory major depression.", 9),
    ("Psychosocial factors including social isolation, trauma history, and chronic illness increase depression risk.", 9),

    # --- Noise / Off-topic (topic -1) ---
    ("Quantum computing uses qubits that exploit superposition and entanglement to perform calculations.", -1),
    ("The Amazon rainforest is experiencing accelerated deforestation driven by agricultural expansion.", -1),
    ("Renaissance art is characterized by humanist themes, perspective, and naturalistic representation.", -1),
    ("Blockchain technology provides decentralized, tamper-resistant ledgers for financial transactions.", -1),
    ("The Big Bang theory describes the origin of the universe from an extremely hot, dense state 13.8 billion years ago.", -1),
    ("Supply chain disruptions in semiconductor manufacturing affect global electronics production.", -1),
    ("Classical music theory encompasses harmony, counterpoint, and form as developed from the Baroque period onward.", -1),
    ("Deep-sea hydrothermal vents host unique ecosystems based on chemosynthesis rather than photosynthesis.", -1),
]

QUERIES_LABELED = [
    ("What is the pharmacological treatment of type 2 diabetes with metformin and alternatives?", 0),
    ("What are the clinical signs and biomarkers of myocardial infarction?", 1),
    ("How does MRSA develop antibiotic resistance and what are treatment options?", 2),
    ("How is pulmonary embolism diagnosed and treated?", 3),
    ("What is the mechanism of action of ACE inhibitors in hypertension?", 4),
    ("What are the modifiable risk factors and treatment of ischemic stroke?", 5),
    ("How effective are COVID-19 mRNA vaccines and what are the concerns?", 6),
    ("How is chronic kidney disease staged and managed?", 7),
    ("What is the pathophysiology and treatment of asthma?", 8),
    ("What are the first-line treatments for major depressive disorder?", 9),
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def average_precision(relevant_mask: np.ndarray) -> float:
    """AP for a ranked list given a boolean relevance mask."""
    hits = np.cumsum(relevant_mask)
    positions = np.arange(1, len(relevant_mask) + 1)
    precisions = hits / positions
    return float(np.sum(precisions * relevant_mask) / max(1, relevant_mask.sum()))


def ndcg_at_k(relevant_mask: np.ndarray, k: int = 10) -> float:
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevant_mask[:k]))
    ideal = sorted(relevant_mask, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_similarity_gap(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    relevance: list[list[int]],
) -> tuple[float, float, float]:
    """Returns (mean_relevant_sim, mean_irrelevant_sim, gap)."""
    all_rel, all_irrel = [], []
    sims = query_embs @ corpus_embs.T  # cosine on normalized vecs
    for qi, rel_ids in enumerate(relevance):
        rel_set = set(rel_ids)
        for ci in range(corpus_embs.shape[0]):
            (all_rel if ci in rel_set else all_irrel).append(sims[qi, ci])
    mr, mi = np.mean(all_rel), np.mean(all_irrel)
    return float(mr), float(mi), float(mr - mi)


def evaluate(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    relevance: list[list[int]],
    k: int = 10,
) -> dict:
    # Normalize for cosine similarity
    q = normalize(query_embs)
    c = normalize(corpus_embs)
    sims = q @ c.T  # (n_queries, n_corpus)

    aps, ndcgs = [], []
    for qi, rel_ids in enumerate(relevance):
        ranked = np.argsort(-sims[qi])
        rel_mask = np.array([1 if r in set(rel_ids) else 0 for r in ranked])
        aps.append(average_precision(rel_mask))
        ndcgs.append(ndcg_at_k(rel_mask, k))

    mr, mi, gap = mean_similarity_gap(q, c, relevance)
    return {
        "MAP": float(np.mean(aps)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "MeanRelSim": mr,
        "MeanIrrelSim": mi,
        "SimGap": gap,
    }


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str], client: OpenAI) -> np.ndarray:
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
    else:
        cache = {}

    results = {}
    to_fetch = [t for t in texts if t not in cache]

    if to_fetch:
        print(f"  Fetching {len(to_fetch)} embeddings from OpenAI...")
        for i in range(0, len(to_fetch), 100):
            batch = to_fetch[i : i + 100]
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            for item, text in zip(resp.data, batch):
                cache[text] = item.embedding
            if i + 100 < len(to_fetch):
                time.sleep(0.5)
        CACHE_FILE.write_text(json.dumps(cache))
        print(f"  Cached to {CACHE_FILE}")

    return np.array([cache[t] for t in texts], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    queries_text = [q for q, _ in QUERIES_LABELED]
    corpus_text  = [c for c, _ in CORPUS_LABELED]
    query_topics = [t for _, t in QUERIES_LABELED]
    corpus_topics = [t for _, t in CORPUS_LABELED]

    # Build relevance: query i is relevant to corpus docs with same topic
    relevance = []
    for qt in query_topics:
        rel_ids = [ci for ci, ct in enumerate(corpus_topics) if ct == qt]
        relevance.append(rel_ids)

    print(f"\n{'='*62}")
    print(f" PCA Embedding Experiment — Medical Domain")
    print(f" Model: {EMBEDDING_MODEL}  |  Queries: {len(queries_text)}  |  Corpus: {len(corpus_text)}")
    print(f" Noise docs (off-topic): {sum(1 for t in corpus_topics if t == -1)}")
    print(f"{'='*62}\n")

    print("Step 1: Fetching embeddings...")
    all_texts = queries_text + corpus_text
    all_embs = get_embeddings(all_texts, client)
    q_embs = all_embs[: len(queries_text)]
    c_embs = all_embs[len(queries_text) :]
    n_dims = q_embs.shape[1]
    print(f"  Embedding dimensions: {n_dims}\n")

    print("Step 2: Evaluating at each PCA dimension...\n")

    # Baseline — full dimensions
    baseline = evaluate(q_embs, c_embs, relevance)
    rows = [{"Dims": n_dims, "Method": "Baseline (no PCA)", **baseline}]

    print(f"  {'Dims':>6}  {'Method':<22}  {'MAP':>6}  {'NDCG@10':>8}  {'RelSim':>7}  {'IrrelSim':>9}  {'Gap':>6}")
    print(f"  {'-'*6}  {'-'*22}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*6}")

    def fmt_row(r):
        return (
            f"  {r['Dims']:>6}  {r['Method']:<22}  "
            f"{r['MAP']:>6.4f}  {r['NDCG@10']:>8.4f}  "
            f"{r['MeanRelSim']:>7.4f}  {r['MeanIrrelSim']:>9.4f}  "
            f"{r['SimGap']:>6.4f}"
        )

    print(fmt_row(rows[0]))

    # Fit PCA on combined query + corpus embeddings (as would be done in practice)
    combined = np.vstack([q_embs, c_embs])

    max_pca = min(combined.shape) - 1  # PCA limit = min(n_samples, n_features) - 1
    print(f"  (PCA max components limited to {max_pca} by corpus size)\n")

    for d in PCA_DIMS:
        if d >= n_dims or d > max_pca:
            continue
        pca = PCA(n_components=d, random_state=42)
        pca.fit(combined)
        q_pca = pca.transform(q_embs)
        c_pca = pca.transform(c_embs)
        metrics = evaluate(q_pca, c_pca, relevance)
        row = {"Dims": d, "Method": f"PCA-{d}", **metrics}
        rows.append(row)
        print(fmt_row(row))

    # Summary
    print(f"\n{'='*62}")
    print(" Summary")
    print(f"{'='*62}")
    df = pd.DataFrame(rows).set_index("Dims")
    best_map = df["MAP"].idxmax()
    best_ndcg = df["NDCG@10"].idxmax()
    best_gap = df["SimGap"].idxmax()
    print(f"  Best MAP:      {best_map} dims  ({df.loc[best_map, 'MAP']:.4f})")
    print(f"  Best NDCG@10:  {best_ndcg} dims  ({df.loc[best_ndcg, 'NDCG@10']:.4f})")
    print(f"  Best SimGap:   {best_gap} dims  ({df.loc[best_gap, 'SimGap']:.4f})")

    baseline_map = rows[0]["MAP"]
    pca_maps = [(r["Dims"], r["MAP"]) for r in rows[1:]]
    improved = [(d, m) for d, m in pca_maps if m > baseline_map]
    if improved:
        print(f"\n  PCA IMPROVED over baseline MAP at: {[d for d,_ in improved]}")
        print(f"  Best PCA MAP: {max(m for _,m in improved):.4f} vs baseline {baseline_map:.4f}")
    else:
        print(f"\n  PCA did NOT improve MAP over the baseline ({baseline_map:.4f}).")
        closest = min(pca_maps, key=lambda x: abs(x[1] - baseline_map))
        print(f"  Closest PCA: {closest[0]} dims (MAP={closest[1]:.4f})")

    # Variance explained
    print(f"\n  Explained variance by PCA dimension:")
    pca_full = PCA(n_components=max_pca, random_state=42)
    pca_full.fit(combined)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    for d in PCA_DIMS:
        if d < n_dims and d <= max_pca:
            print(f"    {d:>5} dims → {cumvar[d-1]*100:.1f}% variance explained")

    out_csv = "pca_results.csv"
    df.reset_index().to_csv(out_csv, index=False)
    print(f"\n  Results saved to {out_csv}\n")


if __name__ == "__main__":
    main()
