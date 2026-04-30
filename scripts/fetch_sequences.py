"""
fetch_sequences.py — Download protein sequences from UniProt for all proteins
in the BRENDA hydrolase dataset.

Saves:
  data/sequences/sequences.fasta        — FASTA file for all UniProt IDs
  data/sequences/sequences.csv          — UniProt ID → sequence mapping (CSV)
  data/sequences/failed_ids.txt         — IDs that could not be fetched

Usage:
  conda run -n MolProtGraphRepresentation python scripts/fetch_sequences.py
  conda run -n MolProtGraphRepresentation python scripts/fetch_sequences.py \
      --batch_size 100 --delay 0.5
"""

import os
import sys
import time
import argparse
import logging
import requests
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
CSV_PATH     = os.path.join(PROJECT_ROOT, 'data', 'dataset_with_decoys.csv')
OUT_DIR      = os.path.join(PROJECT_ROOT, 'data', 'sequences')
os.makedirs(OUT_DIR, exist_ok=True)

UNIPROT_API  = "https://rest.uniprot.org/uniprotkb/{uid}.fasta"


def fetch_batch(ids: list, delay: float = 0.3) -> dict:
    """Fetch sequences for a batch of UniProt IDs. Returns {id: sequence}."""
    results = {}
    for uid in ids:
        try:
            url = UNIPROT_API.format(uid=uid)
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                lines = resp.text.strip().split('\n')
                seq = ''.join(l for l in lines if not l.startswith('>'))
                results[uid] = seq
            else:
                log.warning(f"  {uid}: HTTP {resp.status_code}")
                results[uid] = None
        except Exception as e:
            log.warning(f"  {uid}: {e}")
            results[uid] = None
        time.sleep(delay)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--delay',      type=float, default=0.3,
                        help='Seconds between requests')
    parser.add_argument('--resume',     action='store_true',
                        help='Skip IDs already in sequences.csv')
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)
    all_ids = df['uniprot'].unique().tolist()
    log.info(f"Total unique UniProt IDs: {len(all_ids)}")

    # Resume support
    seq_csv = os.path.join(OUT_DIR, 'sequences.csv')
    done_ids = set()
    if args.resume and os.path.exists(seq_csv):
        prev = pd.read_csv(seq_csv)
        done_ids = set(prev['uniprot'].tolist())
        log.info(f"Resuming: {len(done_ids)} already fetched")

    remaining = [i for i in all_ids if i not in done_ids]
    log.info(f"Fetching {len(remaining)} sequences...")

    all_seqs = {}
    if done_ids and os.path.exists(seq_csv):
        prev = pd.read_csv(seq_csv)
        for _, row in prev.iterrows():
            all_seqs[row['uniprot']] = row['sequence']

    failed = []
    for i in range(0, len(remaining), args.batch_size):
        batch = remaining[i:i + args.batch_size]
        log.info(f"  Batch {i//args.batch_size + 1} / "
                 f"{(len(remaining) + args.batch_size - 1) // args.batch_size} "
                 f"({len(batch)} IDs)...")
        results = fetch_batch(batch, delay=args.delay)
        for uid, seq in results.items():
            if seq:
                all_seqs[uid] = seq
            else:
                failed.append(uid)

        # Save progress after each batch
        rows = [{'uniprot': uid, 'sequence': seq, 'length': len(seq)}
                for uid, seq in all_seqs.items()]
        pd.DataFrame(rows).to_csv(seq_csv, index=False)

    log.info(f"\nDone: {len(all_seqs)} sequences fetched, {len(failed)} failed")

    # Write FASTA
    fasta_path = os.path.join(OUT_DIR, 'sequences.fasta')
    with open(fasta_path, 'w') as f:
        for uid, seq in all_seqs.items():
            f.write(f">{uid}\n{seq}\n")
    log.info(f"FASTA saved: {fasta_path}")

    # Write failed IDs
    if failed:
        failed_path = os.path.join(OUT_DIR, 'failed_ids.txt')
        with open(failed_path, 'w') as f:
            f.write('\n'.join(failed))
        log.info(f"Failed IDs saved: {failed_path}")

    # Summary stats
    lengths = [len(s) for s in all_seqs.values()]
    if lengths:
        import numpy as np
        log.info(f"Sequence length: mean={np.mean(lengths):.0f}, "
                 f"min={min(lengths)}, max={max(lengths)}")


if __name__ == '__main__':
    main()
