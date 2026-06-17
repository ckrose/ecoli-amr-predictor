"""
E. coli AMR Resistance Predictor
=================================
Predicts antibiotic resistance for E. coli genomes using machine learning.

HOW TO USE:
-----------
1. Sequence your E. coli isolate and upload to BV-BRC (bv-brc.org)
   -> Services -> Genome Assembly & Annotation
2. Wait for BV-BRC to assign a Genome ID (looks like: 562.XXXXXX)
3. Run this script:

   Single genome:
       python run_prediction.py 562.86537

   Multiple genomes:
       python run_prediction.py 562.86537 562.193047 562.36249

   From a text file (one genome ID per line):
       python run_prediction.py --file my_genomes.txt

OUTPUT:
-------
- Resistance prediction (R = Resistant, S = Susceptible) + confidence %
- Predicted MIC (minimum inhibitory concentration in mg/L)
- Clinical breakpoint for comparison
- Key AMR genes detected
- Summary saved to predictions_output.txt

ANTIBIOTICS COVERED:
--------------------
- Ampicillin          (breakpoint: 8.0 mg/L)
- Ciprofloxacin       (breakpoint: 0.25 mg/L)
- Gentamicin          (breakpoint: 2.0 mg/L)
- Trimethoprim/Sulfa  (breakpoint: 4.0 mg/L)
- Tetracycline        (breakpoint: 8.0 mg/L)

MODEL PERFORMANCE (5-fold cross-validation on lab-measured data):
-----------------------------------------------------------------
Ampicillin:    89.3% accuracy | AUROC 0.953 | MIC R² 0.903
Ciprofloxacin: 90.0% accuracy | AUROC 0.922 | MIC R² 0.861
Gentamicin:    95.4% accuracy | AUROC 0.904 | MIC R² 0.932
Trimethoprim:  91.3% accuracy | AUROC 0.960 | MIC R² 0.978
Tetracycline:  92.4% accuracy | AUROC 0.954 | MIC R² 0.821
"""

import sys
import argparse
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent

ANTIBIOTICS = {
    "amp": {"name": "Ampicillin",                   "breakpoint": 8.0,   "unit": "mg/L"},
    "cip": {"name": "Ciprofloxacin",                "breakpoint": 0.25,  "unit": "mg/L"},
    "gen": {"name": "Gentamicin",                   "breakpoint": 2.0,   "unit": "mg/L"},
    "tmp": {"name": "Trimethoprim/Sulfamethoxazole", "breakpoint": 4.0,   "unit": "mg/L"},
    "tet": {"name": "Tetracycline",                 "breakpoint": 8.0,   "unit": "mg/L"},
}

MUTATION_GENES = {"gyrA", "gyrB", "parC", "parE"}
SP_GENE_URL = "https://www.bv-brc.org/api/sp_gene"
SELECT_FIELDS = "genome_id,gene,product,property,source,identity,query_coverage,subject_coverage"

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
def load_models() -> dict:
    models = {}
    missing = []
    for ab in ANTIBIOTICS:
        # Classification models saved as model_<ab>.json
        cls_path = DATA_DIR / f"model_{ab}.json"
        if cls_path.exists():
            m = xgb.XGBClassifier()
            m.load_model(cls_path)
            models[f"{ab}_cls"] = m
        else:
            missing.append(f"model_{ab}.json")
        # MIC models saved as model_<ab>_mic.json
        mic_path = DATA_DIR / f"model_{ab}_mic.json"
        if mic_path.exists():
            m = xgb.XGBRegressor()
            m.load_model(mic_path)
            models[f"{ab}_mic"] = m
        else:
            missing.append(f"model_{ab}_mic.json")
    if missing:
        print(f"  Note: missing model files (skipping): {', '.join(missing)}")
    return models


def load_feature_list(ab: str, mtype: str) -> list[str]:
    path = DATA_DIR / f"features_{ab}_{mtype}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature list not found: {path}\n"
            "Please run train_model.py and train_mic_regression.py first."
        )
    return pd.read_csv(path)["0"].tolist()


# ---------------------------------------------------------------------------
# Query BV-BRC
# ---------------------------------------------------------------------------
def query_genes(genome_ids: list[str]) -> pd.DataFrame:
    id_list = ",".join(genome_ids)
    rql = (
        f'and(in(genome_id,({id_list})),eq(property,"Antibiotic Resistance"))'
        f'&limit(25000)&select({SELECT_FIELDS})'
    )
    try:
        r = requests.get(SP_GENE_URL + "?" + rql, headers={"Accept": "application/json"}, timeout=60)
        r.raise_for_status()
        rows = r.json()
    except requests.exceptions.Timeout:
        print("  ERROR: BV-BRC API timed out. Check your internet connection and try again.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Could not reach BV-BRC API: {e}")
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame(columns=["genome_id", "gene", "identity", "product", "source"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build feature vectors
# ---------------------------------------------------------------------------
def build_features(genes_df: pd.DataFrame, genome_ids: list[str], feature_cols: list[str]) -> pd.DataFrame:
    if genes_df.empty:
        missing = {col: 0 for col in feature_cols}
        df = pd.DataFrame({"genome_id": genome_ids})
        return pd.concat([df, pd.DataFrame(missing, index=df.index)], axis=1)

    genes_df = genes_df[genes_df["identity"] >= 80].copy()
    genes_df["genome_id"] = genes_df["genome_id"].astype(str)

    # Binary presence/absence for non-mutation genes
    binary = genes_df[~genes_df["gene"].isin(MUTATION_GENES)]
    binary_matrix = (
        binary.groupby(["genome_id", "gene"]).size()
        .unstack(fill_value=0).clip(upper=1).reset_index()
    )

    # Identity scores for mutation-driven genes
    identity = genes_df[genes_df["gene"].isin(MUTATION_GENES)]
    if not identity.empty:
        identity_matrix = (
            identity.groupby(["genome_id", "gene"])["identity"].max()
            .unstack(fill_value=0)
            .rename(columns={g: f"{g}_identity" for g in MUTATION_GENES})
            .reset_index()
        )
        matrix = binary_matrix.merge(identity_matrix, on="genome_id", how="left")
    else:
        matrix = binary_matrix

    for g in MUTATION_GENES:
        col = f"{g}_identity"
        if col in matrix.columns:
            matrix[col] = matrix[col].fillna(100)  # absent = assume wildtype

    # Ensure all queried genomes present
    all_genomes = pd.DataFrame({"genome_id": genome_ids})
    matrix = all_genomes.merge(matrix, on="genome_id", how="left")

    # Align columns to training feature set
    missing = {col: 0 for col in feature_cols if col not in matrix.columns}
    if missing:
        matrix = pd.concat([matrix, pd.DataFrame(missing, index=matrix.index)], axis=1)
    return matrix[["genome_id"] + feature_cols].fillna(0).copy()


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------
def run_predictions(genome_ids: list[str], models: dict) -> list[dict]:
    print(f"Querying BV-BRC for AMR genes...")
    genes_df = query_genes(genome_ids)
    n_genomes_with_hits = genes_df["genome_id"].nunique() if not genes_df.empty else 0
    print(f"  {len(genes_df)} gene hits across {n_genomes_with_hits}/{len(genome_ids)} genomes\n")

    # Cache feature matrices per model key to avoid rebuilding
    feature_cache = {}
    all_results = []

    for i, gid in enumerate(genome_ids):
        genome_result = {"genome_id": gid, "antibiotics": {}}

        for ab, info in ANTIBIOTICS.items():
            cls_key = f"{ab}_cls"
            mic_key = f"{ab}_mic"

            if cls_key not in models:
                continue

            # Classification
            fc = load_feature_list(ab, "cls")
            if cls_key not in feature_cache:
                feature_cache[cls_key] = (build_features(genes_df, genome_ids, fc), fc)
            mat, fc = feature_cache[cls_key]
            xi = mat[fc].values[i:i+1].astype(np.float32)
            prob_resistant = models[cls_key].predict_proba(xi)[0][1]
            pred_rs = "R" if prob_resistant >= 0.5 else "S"
            confidence = prob_resistant if pred_rs == "R" else 1 - prob_resistant

            # MIC regression
            pred_mic = None
            if mic_key in models:
                fm = load_feature_list(ab, "mic")
                if mic_key not in feature_cache:
                    feature_cache[mic_key] = (build_features(genes_df, genome_ids, fm), fm)
                mat_m, fm = feature_cache[mic_key]
                xi_m = mat_m[fm].values[i:i+1].astype(np.float32)
                log2_mic = models[mic_key].predict(xi_m)[0]
                pred_mic = float(2 ** log2_mic)

            genome_result["antibiotics"][ab] = {
                "name":       info["name"],
                "breakpoint": info["breakpoint"],
                "pred_rs":    pred_rs,
                "confidence": float(confidence),
                "pred_mic":   pred_mic,
            }

        # AMR genes detected
        if not genes_df.empty:
            hits = genes_df[genes_df["genome_id"] == gid]
            genome_result["genes_detected"] = hits["gene"].value_counts().index.tolist()
            genome_result["n_gene_hits"] = len(hits)
        else:
            genome_result["genes_detected"] = []
            genome_result["n_gene_hits"] = 0

        all_results.append(genome_result)

    return all_results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_results(results: list[dict]) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append("  E. COLI AMR RESISTANCE PREDICTION REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 65)

    for res in results:
        gid = res["genome_id"]
        lines.append(f"\nGenome ID: {gid}")
        lines.append(f"AMR gene hits: {res['n_gene_hits']}")
        lines.append("-" * 65)
        lines.append(f"{'Antibiotic':<33} {'R/S':>4}  {'Confidence':>10}  {'Pred MIC':>10}  {'Breakpoint':>10}")
        lines.append("-" * 65)

        resistant_list = []
        for ab, r in res["antibiotics"].items():
            mic_str = f"{r['pred_mic']:.3f} mg/L" if r["pred_mic"] is not None else "N/A"
            bp_str  = f"{r['breakpoint']:.3f} mg/L"
            flag    = " ⚠" if r["pred_rs"] == "R" else ""
            lines.append(
                f"{r['name']:<33} {r['pred_rs']:>4}  {r['confidence']:>9.1%}  "
                f"{mic_str:>10}  {bp_str:>10}{flag}"
            )
            if r["pred_rs"] == "R":
                resistant_list.append(r["name"])

        lines.append("-" * 65)

        if resistant_list:
            lines.append(f"  ⚠  RESISTANT TO: {', '.join(resistant_list)}")
        else:
            lines.append("  ✓  Susceptible to all tested antibiotics")

        if res["genes_detected"]:
            top_genes = res["genes_detected"][:12]
            lines.append(f"  Key AMR genes: {', '.join(top_genes)}")

        lines.append("")

    lines.append("=" * 65)
    lines.append("DISCLAIMER: For research use only. Not for clinical diagnosis.")
    lines.append("Breakpoints per EUCAST 2024 guidelines.")
    lines.append("=" * 65)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Predict E. coli antibiotic resistance from BV-BRC genome IDs."
    )
    parser.add_argument("genome_ids", nargs="*", help="BV-BRC genome ID(s), e.g. 562.86537")
    parser.add_argument("--file", "-f", help="Text file with one genome ID per line")
    parser.add_argument("--output", "-o", default="predictions_output.txt",
                        help="Output file (default: predictions_output.txt)")
    args = parser.parse_args()

    # Collect genome IDs
    genome_ids = list(args.genome_ids)
    if args.file:
        with open(args.file) as f:
            genome_ids += [line.strip() for line in f if line.strip()]
    if not genome_ids:
        parser.print_help()
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  E. COLI AMR PREDICTOR")
    print(f"{'='*55}")
    print(f"  Genomes to predict: {len(genome_ids)}")

    print(f"\nLoading models...")
    models = load_models()
    print(f"  Loaded {len(models)} models ({len(models)//2} antibiotics)")

    results = run_predictions(genome_ids, models)
    report  = format_results(results)

    print(report)

    out_path = DATA_DIR / args.output
    out_path.write_text(report)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
