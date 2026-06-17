"""
Build a gene presence/absence feature matrix for ML training.

Steps:
1. Load phenotype CSVs, filter to Laboratory Method only
2. Load specialty genes parquet
3. Pivot genes into binary features per genome
4. Join with phenotype labels
5. Save train-ready CSV

Output: feature_matrix.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/connorrosenstein/Desktop/ISEF")

# ---------------------------------------------------------------------------
# 1. Load and filter phenotype labels
# ---------------------------------------------------------------------------
def load_phenotype_csv(path, label):
    """Load a phenotype CSV, normalizing column names from either format."""
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={
        "genome_id": "Genome ID",
        "resistant_phenotype": "Resistant Phenotype",
        "evidence": "Evidence",
        "measurement_sign": "Measurement Sign",
        "measurement_value": "Measurement Value",
    })
    df = df[df["Evidence"] == "Laboratory Method"].copy()
    print(f"  {label}: {len(df)} rows | {df['Genome ID'].nunique()} unique genomes")
    return df

print("Loading phenotype data...")
amp     = load_phenotype_csv(DATA_DIR / "BVBRC_genome_amr_ampicillin.csv", "Ampicillin")
cip     = load_phenotype_csv(DATA_DIR / "BVBRC_genome_amr_ciprofloxacin.csv", "Ciprofloxacin")
gen_raw = load_phenotype_csv(DATA_DIR / "BVBRC_genome_amr_gentamicin.csv", "Gentamicin")
tmp_raw = load_phenotype_csv(DATA_DIR / "BVBRC_genome_amr_trimethoprim_sulfamethoxazole.csv", "Trimethoprim/sulfa")
tet_raw = load_phenotype_csv(DATA_DIR / "BVBRC_genome_amr_tetracycline.csv", "Tetracycline")

# Collapse Intermediate -> Resistant (standard in AMR literature)
def clean_phenotype(df):
    df = df.copy()
    df["Resistant Phenotype"] = df["Resistant Phenotype"].replace("Intermediate", "Resistant")
    return df

for df in [amp, cip, gen_raw, tmp_raw, tet_raw]:
    df["Resistant Phenotype"] = df["Resistant Phenotype"].replace("Intermediate", "Resistant")

def deduplicate_phenotype(df, antibiotic):
    df = df[df["Resistant Phenotype"].isin(["Resistant", "Susceptible"])].copy()
    label_col = f"label_{antibiotic}"
    deduped = (
        df.groupby("Genome ID")["Resistant Phenotype"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"Resistant Phenotype": label_col, "Genome ID": "genome_id"})
    )
    deduped[label_col] = (deduped[label_col] == "Resistant").astype(int)
    deduped["genome_id"] = deduped["genome_id"].astype(str)
    print(f"  {label_col}: {deduped[label_col].value_counts().to_dict()}")
    return deduped

amp_labels = deduplicate_phenotype(amp,     "amp")
cip_labels = deduplicate_phenotype(cip,     "cip")
gen_labels = deduplicate_phenotype(gen_raw, "gen")
tmp_labels = deduplicate_phenotype(tmp_raw, "tmp")
tet_labels = deduplicate_phenotype(tet_raw, "tet")

# ---------------------------------------------------------------------------
# 2. Load specialty genes
# ---------------------------------------------------------------------------
print("\nLoading specialty genes...")
genes = pd.read_parquet(DATA_DIR / "specialty_genes_raw.parquet")
print(f"  Total gene rows: {len(genes):,}")
print(f"  Unique genomes with gene data: {genes['genome_id'].nunique():,}")
print(f"  Unique genes: {genes['gene'].nunique():,}")

# Filter to high-confidence hits only (identity >= 80%)
genes = genes[genes["identity"] >= 80].copy()
print(f"  After identity >= 80% filter: {len(genes):,} rows")

# ---------------------------------------------------------------------------
# 3. Build binary gene presence/absence matrix
# ---------------------------------------------------------------------------
print("\nBuilding feature matrix...")

genes["genome_id"] = genes["genome_id"].astype(str)

# For most genes: binary presence/absence
# For mutation-driven resistance genes: use max identity score
# (gyrA/parC/gyrB/parE exist in nearly all E. coli; resistance comes from point
#  mutations that lower identity vs the susceptible reference sequence)
MUTATION_GENES = {"gyrA", "gyrB", "parC", "parE"}

binary_genes = genes[~genes["gene"].isin(MUTATION_GENES)]
binary_matrix = (
    binary_genes.groupby(["genome_id", "gene"])
    .size()
    .unstack(fill_value=0)
    .clip(upper=1)
    .reset_index()
)

identity_genes = genes[genes["gene"].isin(MUTATION_GENES)]
identity_matrix = (
    identity_genes.groupby(["genome_id", "gene"])["identity"]
    .max()
    .unstack(fill_value=0)
    .rename(columns={g: f"{g}_identity" for g in MUTATION_GENES})
    .reset_index()
)

gene_matrix = binary_matrix.merge(identity_matrix, on="genome_id", how="left")
# Missing identity = gene not found = assume wildtype (100% identity to susceptible ref)
for g in MUTATION_GENES:
    col = f"{g}_identity"
    if col in gene_matrix.columns:
        gene_matrix[col] = gene_matrix[col].fillna(100)

print(f"  Feature matrix shape: {gene_matrix.shape}")

# ---------------------------------------------------------------------------
# 4. Join with labels
# ---------------------------------------------------------------------------
print("\nJoining with phenotype labels...")

# Merge all labels — keep any genome that has at least one label
label_cols = ["label_amp", "label_cip", "label_gen", "label_tmp", "label_tet"]
all_labels = amp_labels
for ldf in [cip_labels, gen_labels, tmp_labels, tet_labels]:
    all_labels = all_labels.merge(ldf, on="genome_id", how="outer")
df = gene_matrix.merge(all_labels, on="genome_id", how="inner")

for col in label_cols:
    print(f"  Genomes with {col}: {int(df[col].notna().sum())}")
print(f"  Total genomes in matrix: {len(df)}")

gene_cols = [c for c in df.columns if c not in ["genome_id"] + label_cols]
df[gene_cols] = df[gene_cols].fillna(0).astype(np.int8)

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
out_path = DATA_DIR / "feature_matrix.parquet"
df.to_parquet(out_path, index=False)
print(f"\nSaved feature matrix to {out_path}")
print(f"Shape: {df.shape} ({len(gene_cols)} gene features)")
print(f"\nTop 10 most common genes:\n{genes['gene'].value_counts().head(10).to_string()}")
