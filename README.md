# rna-struct-diff

## Setup

```bash
uv sync
```

Create `.env` in the repo root — nothing else reads the paths, all of `data/` and `src/constants.py` resolve through it:

```bash
export PROJECT_PATH='/absolute/path/to/rna-struct-diff/'
export DATA_PATH='$PROJECT_PATH/data'

# optional: wandb, needed by data/process_data.py and main.py train
export WANDB_PROJECT=''
export WANDB_ENTITY=''

# optional: only for tools/ regeneration
export ETERNAFOLD='$PROJECT_PATH/tools/EternaFold'
export X3DNA='$PROJECT_PATH/tools/x3dna-v2.4'
export PATH="$X3DNA/bin:$PATH"
```

## Dataset

`data/processed.pt` — one entry per **unique sequence**, holding every observed structure of it.
Currently 4409 sequences / 11848 structures. Each entry:

| key | contents |
| --- | --- |
| `sequence` | 4-letter string, non-canonical residues already collapsed to `_` |
| `id_list` | PDB ids of the structures, aligned to `coords_list` |
| `coords_list` | one `(L, 27, 3)` float tensor per structure, raw Angstroms, RNA_ATOMS slot order |
| `sec_struct_list`, `sasa_list`, `rfam_list`, `eq_class_list`, `type_list` | per-structure metadata |
| `rmsds_list` | `{(id_a, id_b): C4' RMSD}` within the entry |
| `cluster_seqid0.8`, `cluster_structsim0.45` | cluster ids (CD-HIT-EST 80%, US-align 45%) |

`data/das_split.pt` — 3-tuple of disjoint index lists into `torch.load` order:
train 4205 / val 100 / test 104. It is the Das et al. (2011) test set carved out first
(riboswitches, aptamers, ribozymes), then a cluster-aware split of the rest.
`data/processed_df.csv` is the flat per-sequence view of `processed.pt`.

Consumers: `Sequence.load_sequences_from_file` (`src/data/sec_utils.py`), then
`RNACoGenerationDataset` + `BatchSampler` (`src/data/datasets.py`).

### Pull the raw input

```bash
# RNASolo2, ~14474 PDB files (this is the archive data/process_data.py reads)
# https://rnasolo.cs.put.poznan.pl/archive
mkdir -p data/raw
# ... unzip the RNASolo2 "All" archive into data/raw

# metadata tables
# nrlist_3.306_4.0A.csv     -> data/    (BGSU non-redundant equivalence classes, RiboGen)
# rnasolo-main-table.csv    -> data/
# RFAM_families_27062023.csv -> data/
```

### Generate `processed.pt`

Needs the two external clustering binaries on `PATH` / at their default paths:
`cd-hit-est` and `~/USalign/qTMclust` (https://zhanggroup.org/US-align/).

```bash
uv run data/process_data.py --no_wandb
```

It loads the three metadata CSVs, reads every `data/raw/*.pdb` through
`src.data.data_utils.pdb_to_tensor`, keeps sequences longer than 10 nt, aligns each
structure of a repeated sequence onto the first one, then clusters and writes
`data/processed.pt` and `data/processed_df.csv`.

### Generate `das_split.pt`

```bash
jupyter lab notebooks/split_das.ipynb
```

Reads `processed.pt` + `processed_df.csv`, excludes sequences > 1000 nt from val/test,
pins the Das list to test, and writes `data/das_split.pt`.

### Training

```bash
uv run main.py train -d ./data/processed.pt --split ./data/das_split.pt
uv run main.py inference -c ./e3nn_checkpoint.pt --length 64 --samples 2 -o ./samples
```

Each run prints per-split size stats before the first epoch (unique sequences,
structures, cropped nodes, residue-length range).
