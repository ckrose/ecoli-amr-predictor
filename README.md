# E. coli AMR Resistance Predictor

**ISEF 2025–2026 | Machine Learning + Wetlab**

An ML pipeline that predicts *E. coli* antibiotic resistance from genomic data. Trained on 8,188 lab-confirmed genomes from the BV-BRC database using XGBoost classifiers and MIC regressors.

## Antibiotics Covered

| Antibiotic | Accuracy | AUROC | MIC R² |
|---|---|---|---|
| Ampicillin | 89.3% | 0.953 | 0.903 |
| Ciprofloxacin | 90.0% | 0.922 | 0.861 |
| Gentamicin | 95.4% | 0.904 | 0.932 |
| Trimethoprim/Sulfa | 91.3% | 0.960 | 0.978 |
| Tetracycline | 92.4% | 0.954 | 0.821 |

*5-fold stratified cross-validation on laboratory-confirmed AMR phenotypes (EUCAST 2024 breakpoints)*

---

## Two Ways to Use

### Option 1: Predict from a BV-BRC Genome ID (fastest)
If you already have a genome annotated on BV-BRC:
```bash
python run_prediction.py 562.86537
python run_prediction.py 562.86537 562.193047 562.36249   # multiple genomes
python run_prediction.py --file my_genomes.txt             # from a text file
```

### Option 2: Predict from Raw Sequencing Data (local pipeline)
If you have Nanopore reads or an assembled FASTA from your own sequencer:
```bash
# From raw Nanopore FASTQ reads (assembles with Flye first, ~10–30 min)
python predict_from_sequence.py --fastq my_reads.fastq --sample MySample

# From an already-assembled FASTA (skips assembly, ~2 min)
python predict_from_sequence.py --fasta my_assembly.fasta --sample MySample
```

**Requirements for local pipeline:**
- [Flye](https://github.com/fenderglass/Flye) assembler (`brew install flye`)
- [NCBI AMRFinderPlus](https://github.com/ncbi/amr) (`conda install -c bioconda ncbi-amrfinderplus`)

### Example Output
```
=================================================================
  AMR PREDICTION REPORT — MySample
  Generated: 2026-06-17 11:45:00
=================================================================

AMR genes detected (4): blaTEM-1, sul1, tetA, gyrA

Antibiotic                         R/S   Confidence    Pred MIC   Breakpoint
-----------------------------------------------------------------
Ampicillin                           R       96.2%   48.000 mg/L   8.000 mg/L ⚠
Ciprofloxacin                        R       87.4%    2.341 mg/L   0.250 mg/L ⚠
Gentamicin                           S       91.3%    0.312 mg/L   2.000 mg/L
Trimethoprim/Sulfamethoxazole        R       88.1%   16.000 mg/L   4.000 mg/L ⚠
Tetracycline                         R       94.7%   32.000 mg/L   8.000 mg/L ⚠
-----------------------------------------------------------------

  ⚠  RESISTANT TO: Ampicillin, Ciprofloxacin, Trimethoprim/Sulfamethoxazole, Tetracycline

DISCLAIMER: For research use only. Not for clinical diagnosis.
```

---

## Project Pipeline (for reproducibility)

The `models/` folder already has trained models ready to use. If you want to retrain from scratch:

### Step 1 — Download phenotype data from BV-BRC
Go to [bv-brc.org](https://www.bv-brc.org/) → *E. coli* genome AMR data → download CSVs for each antibiotic. Save them as:
```
BVBRC_genome_amr_ampicillin.csv
BVBRC_genome_amr_ciprofloxacin.csv
BVBRC_genome_amr_gentamicin.csv
BVBRC_genome_amr_trimethoprim_sulfamethoxazole.csv
BVBRC_genome_amr_tetracycline.csv
```

### Step 2 — Query AMR genes from BV-BRC API
```bash
python query_specialty_genes.py
# Outputs: specialty_genes_raw.parquet
# ~2–3 hours (queries 12,000 genomes in batches)
```

### Step 3 — Build the feature matrix
```bash
python build_feature_matrix.py
# Outputs: feature_matrix.parquet (8,188 genomes × 702 gene features)
```

### Step 4 — Train classifiers
```bash
python train_model.py
# Outputs: model_amp.json, model_cip.json, model_gen.json, model_tmp.json, model_tet.json
#          features_*_cls.csv (feature lists per model)
```

### Step 5 — Train MIC regressors
```bash
python train_mic_regression.py
# Outputs: model_*_mic.json, features_*_mic.csv
```

---

## How It Works

1. **Feature engineering**: Gene presence/absence is extracted from BV-BRC's AMR gene database (sp_gene endpoint). Each genome becomes a binary vector of ~700 AMR genes. For mutation-driven resistance genes (gyrA, gyrB, parC, parE), we use % sequence identity instead of binary presence — point mutations that reduce identity signal resistance.

2. **Classification**: XGBoost classifier per antibiotic predicts R (Resistant) or S (Susceptible). Class imbalance handled via `scale_pos_weight`.

3. **MIC regression**: Separate XGBoost regressor per antibiotic predicts log₂(MIC), then back-transformed to mg/L. Evaluated only on exact measurements (not censored `>` or `<` values).

4. **Local pipeline**: Raw Nanopore reads → Flye de novo assembly → NCBI AMRFinderPlus gene detection → same feature encoding → same models.

---

## File Structure

```
├── run_prediction.py          # Predict from BV-BRC genome ID
├── predict_from_sequence.py   # Predict from local FASTQ or FASTA
├── query_specialty_genes.py   # Step 1: download AMR genes from BV-BRC API
├── build_feature_matrix.py    # Step 2: build gene presence/absence matrix
├── train_model.py             # Step 3: train R/S classifiers
├── train_mic_regression.py    # Step 4: train MIC regressors
├── model_*.json               # Trained XGBoost models (10 total)
├── features_*_cls.csv         # Feature lists for classifiers
├── features_*_mic.csv         # Feature lists for MIC models
└── requirements.txt
```

---

## Install

```bash
git clone https://github.com/YOUR_USERNAME/ecoli-amr-predictor
cd ecoli-amr-predictor
pip install -r requirements.txt
```

*For the local sequencing pipeline, also install Flye and AMRFinderPlus (see above).*

---

*For research use only. Not for clinical diagnosis.*
