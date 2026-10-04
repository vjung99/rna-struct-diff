"""Precompute RF2NA target folds for the val/test splits.

For each split index: write a query-only FASTA, fold it, convert the PDB into our
(L, 27, 3) convention, and store one npz per index. Resumable: existing outputs are
skipped, so re-running after an interrupted job costs nothing.

  uv run python tools/RoseTTAFold2NA/eval/precompute.py --split val
  uv run python tools/RoseTTAFold2NA/eval/precompute.py --split test --msa tier2

tier1 (default): we write the .afa and call predict.py directly.
tier2: we let run_RF2NA.sh build an rMSA-lite MSA; requires hhsuite/blast/hmmer/infernal/
       mafft on PATH and the RNA databases under tools/RoseTTAFold2NA/RNA.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RF2NA = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(RF2NA))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import bridge  # noqa: E402

ENV_PY = os.path.join(RF2NA, "env", "bin", "python")
WEIGHTS = os.path.join(RF2NA, "network", "weights", "RF2NA_apr23.pt")
STUB = os.path.join(RF2NA, "stub", "stub")


def to_fasta_header_string(sequence):
    """Our 5-letter alphabet -> what parse_fasta accepts (ACGU + N)."""
    return "".join(b if b in "ACGU" else "N" for b in sequence)


def fold(seq, tag, workdir, msa_tier, timeout):
    """Run RF2NA once; return (pdb_path, npz_path)."""
    os.makedirs(workdir, exist_ok=True)
    fa = os.path.join(workdir, f"{tag}.fa")
    with open(fa, "w") as fh:
        fh.write(f">{tag}\n{to_fasta_header_string(seq)}\n")
    prefix = os.path.join(workdir, "model")

    if msa_tier == "tier2":
        cmd = [os.path.join(RF2NA, "run_RF2NA.sh"), workdir, f"R:{fa}"]
    else:
        cmd = [
            ENV_PY, os.path.join(RF2NA, "run_cpu.py"),
            "-inputs", f"R:{fa}",
            "-prefix", prefix,
            "-model", WEIGHTS,
            "-db", STUB,
        ]
    subprocess.run(cmd, check=True, timeout=timeout, cwd=os.path.join(RF2NA, "network"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return f"{prefix}_00.pdb", f"{prefix}_00.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test", "train"], required=True)
    ap.add_argument("--msa", choices=["tier1", "tier2"], default="tier1")
    ap.add_argument("--dataset", default=os.path.join(REPO, "data", "processed.pt"))
    ap.add_argument("--das-split", default=os.path.join(REPO, "data", "das_split.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "targets"))
    ap.add_argument("--limit", type=int, default=0, help="first N indices only")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.split)
    work_root = os.path.join(args.out, "_work", args.split)
    os.makedirs(out_dir, exist_ok=True)

    data = torch.load(args.dataset, weights_only=False)
    keys = list(data)
    train_idx, val_idx, test_idx = torch.load(args.das_split, weights_only=False)
    indices = {"train": train_idx, "val": val_idx, "test": test_idx}[args.split]
    if args.limit:
        indices = indices[: args.limit]

    print(f"{args.split}: {len(indices)} indices | msa={args.msa} | out={out_dir}")
    t0 = time.time()
    done = skipped = failed = 0
    for n, i in enumerate(indices):
        dest = os.path.join(out_dir, f"{args.split}_{i:05d}.npz")
        if os.path.exists(dest):
            skipped += 1
            continue
        entry = data[keys[i]]
        seq = entry["sequence"]
        workdir = os.path.join(work_root, f"{args.split}_{i:05d}")
        try:
            pdb, npz = fold(seq, f"{args.split}_{i:05d}", workdir, args.msa, args.timeout)
            tokens, x, v, mask = bridge.read_pdb(pdb)
            aux = np.load(npz)
            bridge.save(
                dest, tokens, x, v, mask,
                source_index=np.int64(i),
                source_ids=np.array(entry["id_list"]),
                plddt_mean=np.float32(float(np.nanmean(aux["lddt"]))),
                lddt=aux["lddt"],
                pae=aux["pae"].astype(np.float16),
            )
            done += 1
            rate = (time.time() - t0) / max(done, 1)
            eta = rate * (len(indices) - n - 1) / 60
            print(f"[{n+1}/{len(indices)}] {seq[:20]}... L={len(tokens)} "
                  f"plddt={float(np.nanmean(aux['lddt'])):.3f} eta={eta:.0f}m", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            print(f"[{n+1}/{len(indices)}] index {i} FAILED: {exc}", flush=True)

    manifest = {
        "split": args.split,
        "msa": args.msa,
        "weights": os.path.basename(WEIGHTS),
        "weights_bytes": os.path.getsize(WEIGHTS) if os.path.exists(WEIGHTS) else None,
        "n_done": done, "n_skipped": skipped, "n_failed": failed,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
