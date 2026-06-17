import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import requests
import subprocess
import tempfile
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent

ANTIBIOTICS = {
    "amp": {"name": "Ampicillin",                    "breakpoint": 8.0},
    "cip": {"name": "Ciprofloxacin",                 "breakpoint": 0.25},
    "gen": {"name": "Gentamicin",                    "breakpoint": 2.0},
    "tmp": {"name": "Trimethoprim/Sulfamethoxazole",  "breakpoint": 4.0},
    "tet": {"name": "Tetracycline",                  "breakpoint": 8.0},
}
MUTATION_GENES = {"gyrA", "gyrB", "parC", "parE"}
SP_GENE_URL    = "https://www.bv-brc.org/api/sp_gene"
SELECT_FIELDS  = "genome_id,gene,product,property,source,identity,query_coverage,subject_coverage"
FLYE_BIN       = shutil.which("flye")      or "/opt/homebrew/bin/flye"
AMRFINDER_BIN  = shutil.which("amrfinder") or "/opt/miniconda3/bin/amrfinder"

# ── helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for ab in ANTIBIOTICS:
        for suffix, cls in [("", xgb.XGBClassifier), ("_mic", xgb.XGBRegressor)]:
            p = DATA_DIR / f"model_{ab}{suffix}.json"
            if p.exists():
                m = cls(); m.load_model(p)
                models[f"{ab}{'_cls' if suffix=='' else suffix}"] = m
    return models

def load_features(ab, mtype):
    return pd.read_csv(DATA_DIR / f"features_{ab}_{mtype}.csv")["0"].tolist()

def query_bvbrc(genome_ids):
    rql = (f'and(in(genome_id,({",".join(genome_ids)})),eq(property,"Antibiotic Resistance"))'
           f'&limit(25000)&select({SELECT_FIELDS})')
    r = requests.get(SP_GENE_URL + "?" + rql, headers={"Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    rows = r.json()
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["genome_id","gene","identity"])

def build_features_bvbrc(genes_df, genome_ids, feature_cols):
    if genes_df.empty:
        return pd.DataFrame(0, index=range(len(genome_ids)), columns=["genome_id"]+feature_cols).assign(genome_id=genome_ids)
    genes_df = genes_df[genes_df["identity"] >= 80].copy()
    genes_df["genome_id"] = genes_df["genome_id"].astype(str)
    binary = genes_df[~genes_df["gene"].isin(MUTATION_GENES)]
    mat = binary.groupby(["genome_id","gene"]).size().unstack(fill_value=0).clip(upper=1).reset_index()
    id_df = genes_df[genes_df["gene"].isin(MUTATION_GENES)]
    if not id_df.empty:
        imat = id_df.groupby(["genome_id","gene"])["identity"].max().unstack(fill_value=0)\
                    .rename(columns={g:f"{g}_identity" for g in MUTATION_GENES}).reset_index()
        mat = mat.merge(imat, on="genome_id", how="left")
    for g in MUTATION_GENES:
        col = f"{g}_identity"
        if col in mat.columns: mat[col] = mat[col].fillna(100)
    mat = pd.DataFrame({"genome_id": genome_ids}).merge(mat, on="genome_id", how="left")
    missing = {c:0 for c in feature_cols if c not in mat.columns}
    if missing: mat = pd.concat([mat, pd.DataFrame(missing, index=mat.index)], axis=1)
    return mat[["genome_id"]+feature_cols].fillna(0)

def build_features_local(genes_df, feature_cols):
    if "element_type" in genes_df.columns:
        genes_df = genes_df[genes_df["element_type"]=="AMR"].copy()
    if "identity" in genes_df.columns:
        genes_df = genes_df[genes_df["identity"]>=80].copy()
    row = {}
    for gene in genes_df[~genes_df["gene"].isin(MUTATION_GENES)]["gene"].unique():
        row[gene] = 1
    for g in MUTATION_GENES:
        col = f"{g}_identity"
        m = genes_df[genes_df["gene"]==g]
        row[col] = float(m["identity"].max()) if not m.empty and "identity" in m.columns else 100.0
    return np.array([[row.get(c,0) for c in feature_cols]], dtype=np.float32)

def run_amrfinder(fasta_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "amrfinder.tsv"
    r = subprocess.run([AMRFINDER_BIN, "--nucleotide", str(fasta_path),
                        "--organism","Escherichia","--output",str(tsv),"--threads","4"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not tsv.exists():
        return pd.DataFrame(columns=["gene","identity","element_type"])
    df = pd.read_csv(tsv, sep="\t")
    return df.rename(columns={"Gene symbol":"gene","% Identity":"identity","Element type":"element_type"})

def predict_single(X, ab, models, mtype_suffix="cls"):
    key = f"{ab}_cls"
    prob = models[key].predict_proba(X)[0][1]
    pred = "Resistant" if prob >= 0.5 else "Susceptible"
    conf = prob if pred=="Resistant" else 1-prob
    mic = None
    if f"{ab}_mic" in models:
        fm = load_features(ab, "mic")
        if isinstance(X, np.ndarray):
            Xm = X  # local path: feature lists same length assumption — rebuild below
        mic = float(2 ** models[f"{ab}_mic"].predict(X)[0])
    return pred, conf, mic

def build_results_df(preds):
    rows = []
    for ab, info in ANTIBIOTICS.items():
        pred, conf, mic = preds[ab]
        rows.append({
            "Antibiotic": info["name"],
            "Result": pred,
            "Confidence": f"{conf:.0%}",
            "Predicted MIC": f"{mic:.3f} mg/L" if mic else "—",
            "Breakpoint": f"{info['breakpoint']} mg/L",
        })
    return pd.DataFrame(rows)

# ── page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="E. coli AMR Predictor", layout="centered")
st.title("E. coli AMR Predictor")
st.caption("Predicts antibiotic resistance from genomic data using XGBoost. For research use only.")

models = load_models()

text_color = "white" if st.get_option("theme.base") == "dark" else "black"

tab1, tab2 = st.tabs(["From BV-BRC Genome ID", "From Sequencing File"])

# ── Tab 1: BV-BRC ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Predict from BV-BRC Genome ID")
    genome_input = st.text_input("Enter genome ID(s), separated by spaces or commas",
                                  placeholder="e.g. 562.86537  562.193047")
    if st.button("Run Prediction", key="bvbrc_btn", type="primary"):
        ids = [x.strip() for x in genome_input.replace(","," ").split() if x.strip()]
        if not ids:
            st.warning("Please enter at least one genome ID.")
        else:
            with st.spinner("Querying BV-BRC and running models..."):
                try:
                    genes_df = query_bvbrc(ids)
                    for gid in ids:
                        st.markdown(f"#### Genome `{gid}`")
                        preds = {}
                        for ab in ANTIBIOTICS:
                            fc = load_features(ab, "cls")
                            mat = build_features_bvbrc(genes_df, ids, fc)
                            i   = ids.index(gid)
                            X   = mat[fc].values[i:i+1].astype(np.float32)
                            prob = models[f"{ab}_cls"].predict_proba(X)[0][1]
                            pred = "Resistant" if prob>=0.5 else "Susceptible"
                            conf = prob if pred=="Resistant" else 1-prob
                            mic  = None
                            if f"{ab}_mic" in models:
                                fm = load_features(ab, "mic")
                                Xm = build_features_bvbrc(genes_df, ids, fm)[fm].values[i:i+1].astype(np.float32)
                                mic = float(2**models[f"{ab}_mic"].predict(Xm)[0])
                            preds[ab] = (pred, conf, mic)
                        df = build_results_df(preds)
                        def style_row(row):
                            color = "#8b0000" if row["Result"]=="Resistant" else "#1a5c1a"
                            return [f"background-color: {color}; color: {text_color}"]*len(row)
                        st.dataframe(df.style.apply(style_row, axis=1), hide_index=True, use_container_width=True)
                        if not genes_df.empty:
                            hits = [g for g in genes_df[genes_df["genome_id"]==gid]["gene"].tolist() if isinstance(g, str)]
                            if hits:
                                st.caption(f"AMR genes detected: {', '.join(hits[:15])}")
                        st.divider()
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 2: Local file ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Predict from Your Sequencing Data")
    st.markdown("Upload a **FASTA** (assembled genome) or **FASTQ** (raw Nanopore reads).")
    uploaded = st.file_uploader("Choose file", type=["fasta","fa","fna","fastq","fq"])
    sample_name = st.text_input("Sample name", placeholder="e.g. Isolate_1")

    if st.button("Run Prediction", key="local_btn", type="primary"):
        if not uploaded:
            st.warning("Please upload a file.")
        else:
            with st.spinner("Running AMRFinderPlus and models..."):
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp = Path(tmp)
                        suffix = Path(uploaded.name).suffix
                        in_path = tmp / f"input{suffix}"
                        in_path.write_bytes(uploaded.read())

                        # Assembly if FASTQ
                        if suffix.lower() in (".fastq",".fq"):
                            st.info("Assembling reads with Flye (this takes 10–30 min)...")
                            asm_dir = tmp / "assembly"
                            r = subprocess.run([FLYE_BIN,"--nano-raw",str(in_path),
                                                "--out-dir",str(asm_dir),"--threads","4",
                                                "--genome-size","5m"],
                                               capture_output=True, text=True)
                            if r.returncode != 0:
                                st.error(f"Assembly failed:\n{r.stderr[-1000:]}")
                                st.stop()
                            fasta_path = asm_dir / "assembly.fasta"
                        else:
                            fasta_path = in_path

                        genes_df = run_amrfinder(fasta_path, tmp/"amr")
                        st.success(f"Found {len(genes_df)} AMR gene hits")

                        preds = {}
                        for ab in ANTIBIOTICS:
                            fc = load_features(ab, "cls")
                            X  = build_features_local(genes_df, fc)
                            prob = models[f"{ab}_cls"].predict_proba(X)[0][1]
                            pred = "Resistant" if prob>=0.5 else "Susceptible"
                            conf = prob if pred=="Resistant" else 1-prob
                            mic  = None
                            if f"{ab}_mic" in models:
                                fm = load_features(ab, "mic")
                                Xm = build_features_local(genes_df, fm)
                                mic = float(2**models[f"{ab}_mic"].predict(Xm)[0])
                            preds[ab] = (pred, conf, mic)

                        label = sample_name or uploaded.name
                        st.markdown(f"#### {label}")
                        df = build_results_df(preds)
                        def style_row(row):
                            color = "#8b0000" if row["Result"]=="Resistant" else "#1a5c1a"
                            return [f"background-color: {color}; color: {text_color}"]*len(row)
                        st.dataframe(df.style.apply(style_row, axis=1), hide_index=True, use_container_width=True)
                        if not genes_df.empty:
                            amr_hits = genes_df[genes_df.get("element_type", pd.Series(["AMR"]*len(genes_df)))=="AMR"]["gene"].tolist()
                            if amr_hits:
                                st.caption(f"AMR genes detected: {', '.join(amr_hits[:15])}")
                except Exception as e:
                    st.error(f"Error: {e}")

st.divider()
st.caption("For research use only · Not for clinical diagnosis · EUCAST 2024 breakpoints")
