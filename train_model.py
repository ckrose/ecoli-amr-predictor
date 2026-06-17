"""
Train AMR resistance prediction models for ampicillin and ciprofloxacin.

For each antibiotic:
- Trains an XGBoost classifier (gene presence/absence -> R/S prediction)
- Evaluates with 5-fold cross-validation
- Reports accuracy, AUROC, sensitivity, specificity
- Saves the trained model and feature importance

Output:
    model_amp.json       - trained ampicillin model
    model_cip.json       - trained ciprofloxacin model
    results.txt          - evaluation summary
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import xgboost as xgb

DATA_DIR = Path("/Users/connorrosenstein/Desktop/ISEF")

# ---------------------------------------------------------------------------
# Load feature matrix
# ---------------------------------------------------------------------------
print("Loading feature matrix...")
df = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")
LABEL_COLS = ["label_amp", "label_cip", "label_gen", "label_tmp", "label_tet"]
gene_cols = [c for c in df.columns if c not in ["genome_id"] + LABEL_COLS]
print(f"  {len(df)} genomes, {len(gene_cols)} gene features")

results_lines = []

def train_and_evaluate(antibiotic: str, label_col: str):
    print(f"\n{'='*50}")
    print(f"Training {antibiotic} model...")

    # Drop rows with no label for this antibiotic
    subset = df[df[label_col].notna()].copy()
    X = subset[gene_cols].values.astype(np.float32)
    y = subset[label_col].values.astype(int)

    n_resistant = y.sum()
    n_susceptible = len(y) - n_resistant
    print(f"  Samples: {len(y)} ({n_resistant} resistant, {n_susceptible} susceptible)")

    # Scale pos weight to handle class imbalance
    scale_pos_weight = n_susceptible / n_resistant

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = {"accuracy": [], "auroc": [], "sensitivity": [], "specificity": []}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        auroc = roc_auc_score(y_val, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        sensitivity = tp / (tp + fn)  # recall for resistant class
        specificity = tn / (tn + fp)  # recall for susceptible class

        fold_results["accuracy"].append(acc)
        fold_results["auroc"].append(auroc)
        fold_results["sensitivity"].append(sensitivity)
        fold_results["specificity"].append(specificity)

        print(f"  Fold {fold+1}: acc={acc:.3f} auroc={auroc:.3f} sens={sensitivity:.3f} spec={specificity:.3f}")

    # Summary
    print(f"\n  === {antibiotic.upper()} RESULTS (5-fold CV) ===")
    for metric, values in fold_results.items():
        mean, std = np.mean(values), np.std(values)
        print(f"  {metric:12s}: {mean:.3f} ± {std:.3f}")
        results_lines.append(f"{antibiotic} {metric}: {mean:.3f} ± {std:.3f}")

    # Train final model on all data
    model.fit(X, y, verbose=False)
    model.save_model(DATA_DIR / f"model_{antibiotic}.json")
    pd.Series(gene_cols).to_csv(DATA_DIR / f"features_{antibiotic}_cls.csv", index=False)
    print(f"  Model saved to model_{antibiotic}.json")

    # Top 20 most important genes
    importances = pd.Series(model.feature_importances_, index=gene_cols)
    top_genes = importances.nlargest(20)
    print(f"\n  Top 20 predictive genes:")
    print(top_genes.to_string())
    results_lines.append(f"\n{antibiotic} top genes:\n{top_genes.to_string()}\n")

    return model


model_amp = train_and_evaluate("amp", "label_amp")
model_cip = train_and_evaluate("cip", "label_cip")
model_gen = train_and_evaluate("gen", "label_gen")
model_tmp = train_and_evaluate("tmp", "label_tmp")
model_tet = train_and_evaluate("tet", "label_tet")

# Save results summary
out = DATA_DIR / "results.txt"
out.write_text("\n".join(results_lines))
print(f"\nResults saved to {out}")
