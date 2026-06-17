"""
Batch query BV-BRC sp_gene API for antibiotic resistance specialty genes.

For each genome ID in the AMR phenotype CSVs, fetches rows from the sp_gene
table where property == 'Antibiotic Resistance', then saves a flat parquet
file for downstream feature matrix construction.

Usage:
    python query_specialty_genes.py [--resume] [--batch-size 200] [--workers 8]

Outputs:
    specialty_genes_raw.parquet   – all rows returned by the API
    query_progress.json           – checkpoint so --resume can skip done batches
"""

import argparse
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "https://www.bv-brc.org/api/sp_gene"
DATA_DIR = Path("/Users/connorrosenstein/Desktop/ISEF")
AMP_CSV = DATA_DIR / "BVBRC_genome_amr.csv"
CIP_CSV = DATA_DIR / "BVBRC_genome_amr (1).csv"
OUT_PARQUET = DATA_DIR / "specialty_genes_raw.parquet"
PROGRESS_FILE = DATA_DIR / "query_progress.json"

# Fields to retrieve — keeps payloads small
SELECT_FIELDS = "genome_id,gene,product,property,source,classification,identity,e_value,query_coverage,subject_coverage"

HEADERS = {"Accept": "application/json"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_genome_ids() -> list[str]:
    amp = pd.read_csv(AMP_CSV, usecols=["Genome ID", "Evidence"], low_memory=False)
    cip = pd.read_csv(CIP_CSV, usecols=["Genome ID", "Evidence"], low_memory=False)
    # Only query genomes with real lab-measured phenotypes
    amp = amp[amp["Evidence"] == "Laboratory Method"]
    cip = cip[cip["Evidence"] == "Laboratory Method"]
    ids = (
        pd.concat([amp["Genome ID"], cip["Genome ID"]])
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    log.info("Loaded %d unique genome IDs with lab-measured phenotypes", len(ids))
    return ids


def query_batch(genome_ids: list[str], retries: int = 5) -> list[dict]:
    """GET a single batch via BV-BRC RQL; returns list of result dicts."""
    id_list = ",".join(genome_ids)
    # property value must be quoted in RQL: eq(property,"Antibiotic Resistance")
    # limit(25000) is the BV-BRC hard cap per request — keep batch_size ≤ 50 to avoid truncation
    rql = (
        f'and(in(genome_id,({id_list})),eq(property,"Antibiotic Resistance"))'
        f'&limit(25000)&select({SELECT_FIELDS})'
    )
    backoff = 2
    for attempt in range(retries):
        try:
            # BV-BRC expects raw RQL as the query string; passing via params= would
            # double-encode it, so build the URL manually.
            resp = requests.get(API_URL + "?" + rql, headers=HEADERS, timeout=120)
            resp.raise_for_status()
            docs = resp.json()
            if len(docs) == 25000:
                log.warning(
                    "Batch hit 25k row cap — some genomes may be truncated. "
                    "Reduce --batch-size if this happens frequently."
                )
            return docs
        except Exception as exc:
            if attempt == retries - 1:
                log.error("Batch failed after %d attempts: %s", retries, exc)
                return []
            log.warning("Attempt %d failed (%s); retrying in %ds", attempt + 1, exc, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    return []


def chunked(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip already-done batches")
    parser.add_argument("--limit", type=int, default=None, help="Only process this many genome IDs (for testing)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Genome IDs per API call (keep ≤ 50; BV-BRC caps at 25k rows)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel threads")
    args = parser.parse_args()

    genome_ids = load_genome_ids()
    if args.limit:
        genome_ids = genome_ids[:args.limit]
        log.info("Limited to %d genome IDs for testing", len(genome_ids))
    batches = list(chunked(genome_ids, args.batch_size))
    log.info("Split into %d batches of ≤%d IDs", len(batches), args.batch_size)

    # Load checkpoint
    progress: dict = {}
    if args.resume and PROGRESS_FILE.exists():
        progress = json.loads(PROGRESS_FILE.read_text())
        done = sum(1 for v in progress.values() if v == "done")
        log.info("Resuming: %d/%d batches already done", done, len(batches))

    all_rows: list[dict] = []

    if args.resume and OUT_PARQUET.exists():
        existing = pd.read_parquet(OUT_PARQUET)
        all_rows = existing.to_dict("records")
        log.info("Loaded %d existing rows from parquet", len(all_rows))

    def process(idx_batch):
        idx, batch = idx_batch
        if progress.get(str(idx)) == "done":
            return idx, []
        rows = query_batch(batch)
        return idx, rows

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, (i, b)): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            idx, rows = future.result()
            all_rows.extend(rows)
            progress[str(idx)] = "done"
            completed += 1
            if completed % 5 == 0 or completed == len(batches):
                df = pd.DataFrame(all_rows)
                df.to_parquet(OUT_PARQUET, index=False)
                PROGRESS_FILE.write_text(json.dumps(progress))
                log.info(
                    "Progress: %d/%d batches | %d gene rows accumulated",
                    completed, len(batches), len(all_rows),
                )

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_PARQUET, index=False)
    PROGRESS_FILE.write_text(json.dumps(progress))
    log.info("Done. %d total rows saved to %s", len(df), OUT_PARQUET)
    if not df.empty:
        log.info("Unique genomes with ≥1 AMR gene hit: %d / %d", df["genome_id"].nunique(), len(genome_ids))
        log.info("Top genes:\n%s", df["gene"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
