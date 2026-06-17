"""
Local AMR Predictor — goes from raw sequencer output to resistance prediction
without needing a BV-BRC genome ID.

Pipeline:
    FASTQ (Nanopore reads) → Flye assembler → FASTA → AMRFinderPlus → Prediction

Usage:
    # From raw Nanopore FASTQ reads (assembles first):
    python predict_from_sequence.py --fastq my_reads.fastq --sample MySample

    # From already-assembled FASTA (skip assembly):
    python predict_from_sequence.py --fasta my_assembly.fasta --sample MySample

    # Multiple samples at once:
    python predict_from_sequence.py --fastq sample1.fastq --sample Sample1
    python predict_from_sequence.py --fastq sample2.fastq --sample Sample2

Requirements (already installed):
    - flye         (assembler for Nanopore reads)
    - amrfinder    (NCBI AMRFinderPlus for gene detection)
    - xgboost, pandas, numpy (Python packages)
"""

import argparse
import subprocess
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent

FLYE_BIN      = shutil.which("flye")      or "/opt/homebrew/bin/flye"
AMRFINDER_BIN = shutil.which("amrfinder") or "/opt/miniconda3/bin/amrfinder"

ANTIBIOTICS = {
    "amp": {"name": "Ampicillin",                   "breakpoint": 8.0},
    "cip": {"name": "Ciprofloxacin",                "breakpoint": 0.25},
    "gen": {"name": "Gentamicin",                   "breakpoint": 2.0},
    "tmp": {"name": "Trimethoprim/Sulfamethoxazole", "breakpoint": 4.0},
    "tet": {"name": "Tetracycline",                 "breakpoint": 8.0},
}

MUTATION_GENES = {"gyrA", "gyrB", "parC", "parE"}

# ---------------------------------------------------------------------------
# Step 1: Assemble FASTQ → FASTA with Flye
# ---------------------------------------------------------------------------
def assemble(fastq_path: Path, out_dir: Path, threads: int = 4) -> Path:
    print(f"\n[1/3] Assembling reads with Flye...")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        FLYE_BIN,
        "--nano-raw", str(fastq_path),   # Nanopore raw reads
        "--out-dir",  str(out_dir),
        "--threads",  str(threads),
        "--genome-size", "5m",           # E. coli genome ~5 Mb
        "--min-overlap", "1000",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Flye failed:\n{result.stderr[-2000:]}")
        sys.exit(1)

    fasta = out_dir / "assembly.fasta"
    if not fasta.exists():
        print("  ERROR: Flye finished but assembly.fasta not found.")
        sys.exit(1)

    # Quick stats
    seq_len = sum(len(l.strip()) for l in open(fasta) if not l.startswith(">"))
    n_contigs = sum(1 for l in open(fasta) if l.startswith(">"))
    print(f"  Assembly complete: {n_contigs} contigs, {seq_len/1e6:.2f} Mb total")
    return fasta


# ---------------------------------------------------------------------------
# Step 2: Detect AMR genes with AMRFinderPlus
# ---------------------------------------------------------------------------
def run_amrfinder(fasta_path: Path, out_dir: Path) -> pd.DataFrame:
    print(f"\n[2/3] Detecting AMR genes with AMRFinderPlus...")
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_out = out_dir / "amrfinder_output.tsv"
    cmd = [
        AMRFINDER_BIN,
        "--nucleotide", str(fasta_path),
        "--organism", "Escherichia",      # E. coli specific database
        "--output", str(tsv_out),
        "--plus",                          # include stress/virulence genes too
        "--threads", "4",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: AMRFinderPlus failed:\n{result.stderr[-2000:]}")
        sys.exit(1)

    if not tsv_out.exists() or tsv_out.stat().st_size == 0:
        print("  No AMR genes detected (empty output).")
        return pd.DataFrame(columns=["gene", "identity", "element_type"])

    df = pd.read_csv(tsv_out, sep="\t")
    # AMRFinderPlus columns: Name, Sequence name, Start, Stop, Strand,
    # Gene symbol, Sequence name, Scope, Type, Subtype, % Identity, % Coverage...
    print(f"  Found {len(df)} AMR gene hits")

    # Normalize to match BV-BRC column names our model expects
    col_map = {
        "Gene symbol":   "gene",
        "% Identity":    "identity",
        "% Coverage of reference sequence": "query_coverage",
        "Element type":  "element_type",
        "Element subtype": "element_subtype",
        "Name":          "product",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "identity" not in df.columns and "% Identity to reference sequence" in df.columns:
        df = df.rename(columns={"% Identity to reference sequence": "identity"})

    return df


# ---------------------------------------------------------------------------
# Step 3: Build features from AMRFinderPlus output
# ---------------------------------------------------------------------------
def build_features_from_amrfinder(genes_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Convert AMRFinderPlus output into the same feature vector our model expects."""
    genes_df = genes_df.copy()

    # Filter to resistance genes only (not virulence/stress)
    if "element_type" in genes_df.columns:
        genes_df = genes_df[genes_df["element_type"] == "AMR"].copy()

    # Filter to high-confidence hits
    if "identity" in genes_df.columns:
        genes_df = genes_df[genes_df["identity"] >= 80].copy()

    feature_row = {}

    # Binary presence/absence
    binary_genes = genes_df[~genes_df["gene"].isin(MUTATION_GENES)]
    for gene in binary_genes["gene"].unique():
        feature_row[gene] = 1

    # Identity scores for mutation genes
    for g in MUTATION_GENES:
        col = f"{g}_identity"
        matches = genes_df[genes_df["gene"] == g]
        if not matches.empty and "identity" in matches.columns:
            feature_row[col] = float(matches["identity"].max())
        else:
            feature_row[col] = 100.0  # absent = assume wildtype

    # Align to training feature columns
    X = np.array([[feature_row.get(col, 0) for col in feature_cols]], dtype=np.float32)
    return X


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
def load_models() -> dict:
    models = {}
    for ab in ANTIBIOTICS:
        cls_path = DATA_DIR / f"model_{ab}.json"
        mic_path = DATA_DIR / f"model_{ab}_mic.json"
        if cls_path.exists():
            m = xgb.XGBClassifier(); m.load_model(cls_path)
            models[f"{ab}_cls"] = m
        if mic_path.exists():
            m = xgb.XGBRegressor(); m.load_model(mic_path)
            models[f"{ab}_mic"] = m
    return models


def load_feature_list(ab: str, mtype: str) -> list[str]:
    path = DATA_DIR / f"features_{ab}_{mtype}.csv"
    return pd.read_csv(path)["0"].tolist()


# ---------------------------------------------------------------------------
# Format and print results
# ---------------------------------------------------------------------------
def format_and_print(sample_name: str, genes_df: pd.DataFrame, models: dict) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append(f"  AMR PREDICTION REPORT — {sample_name}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 65)

    amr_genes = genes_df[genes_df.get("element_type", pd.Series(["AMR"]*len(genes_df))) == "AMR"]["gene"].tolist() if not genes_df.empty else []
    lines.append(f"\nAMR genes detected ({len(amr_genes)}): {', '.join(amr_genes[:15]) or 'None'}\n")
    lines.append(f"{'Antibiotic':<33} {'R/S':>4}  {'Confidence':>10}  {'Pred MIC':>10}  {'Breakpoint':>10}")
    lines.append("-" * 65)

    resistant_list = []
    for ab, info in ANTIBIOTICS.items():
        cls_key = f"{ab}_cls"
        mic_key = f"{ab}_mic"
        if cls_key not in models:
            continue

        fc = load_feature_list(ab, "cls")
        X_cls = build_features_from_amrfinder(genes_df, fc)
        prob = models[cls_key].predict_proba(X_cls)[0][1]
        pred_rs = "R" if prob >= 0.5 else "S"
        confidence = prob if pred_rs == "R" else 1 - prob

        mic_str = "N/A"
        if mic_key in models:
            fm = load_feature_list(ab, "mic")
            X_mic = build_features_from_amrfinder(genes_df, fm)
            pred_mic = 2 ** models[mic_key].predict(X_mic)[0]
            mic_str = f"{pred_mic:.3f} mg/L"

        bp = info["breakpoint"]
        flag = " ⚠" if pred_rs == "R" else ""
        lines.append(f"{info['name']:<33} {pred_rs:>4}  {confidence:>9.1%}  {mic_str:>10}  {bp:>7.3f} mg/L{flag}")
        if pred_rs == "R":
            resistant_list.append(info["name"])

    lines.append("-" * 65)
    if resistant_list:
        lines.append(f"\n  ⚠  RESISTANT TO: {', '.join(resistant_list)}")
    else:
        lines.append(f"\n  ✓  Susceptible to all tested antibiotics")

    lines.append("\n" + "=" * 65)
    lines.append("DISCLAIMER: For research use only. Not for clinical diagnosis.")
    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Predict E. coli AMR from raw Nanopore reads or assembled FASTA."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fastq", "-q", type=Path, help="Raw Nanopore FASTQ reads")
    group.add_argument("--fasta", "-f", type=Path, help="Already-assembled FASTA genome")
    parser.add_argument("--sample", "-s", default="Sample", help="Sample name for output")
    parser.add_argument("--threads", "-t", type=int, default=4, help="CPU threads")
    parser.add_argument("--keep-files", action="store_true", help="Keep intermediate files")
    args = parser.parse_args()

    work_dir = DATA_DIR / f"workdir_{args.sample}"

    print(f"\n{'='*55}")
    print(f"  E. COLI LOCAL AMR PREDICTOR")
    print(f"  Sample: {args.sample}")
    print(f"{'='*55}")

    # Step 1: Assembly (if FASTQ provided)
    if args.fastq:
        if not args.fastq.exists():
            print(f"ERROR: FASTQ file not found: {args.fastq}")
            sys.exit(1)
        fasta_path = assemble(args.fastq, work_dir / "assembly", args.threads)
    else:
        if not args.fasta.exists():
            print(f"ERROR: FASTA file not found: {args.fasta}")
            sys.exit(1)
        fasta_path = args.fasta
        print(f"\n[1/3] Skipping assembly (FASTA provided: {fasta_path})")

    # Step 2: AMR gene detection
    genes_df = run_amrfinder(fasta_path, work_dir / "amrfinder")

    # Step 3: Predict
    print(f"\n[3/3] Running resistance predictions...")
    models = load_models()
    print(f"  Loaded {len(models)} models")

    report = format_and_print(args.sample, genes_df, models)
    print("\n" + report)

    # Save report
    out_path = DATA_DIR / f"predictions_{args.sample}.txt"
    out_path.write_text(report)
    print(f"\nReport saved to: {out_path}")

    # Save AMRFinderPlus raw output too
    if not genes_df.empty:
        genes_out = DATA_DIR / f"amr_genes_{args.sample}.tsv"
        genes_df.to_csv(genes_out, sep="\t", index=False)
        print(f"AMR genes saved to: {genes_out}")

    # Cleanup
    if not args.keep_files and work_dir.exists():
        import shutil
        shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
