# Paratope scoring and mutation prioritisation for anti-GREM antibodies

Code and data to reproduce every figure and table of the manuscript. Two
antibodies are analysed against both Gremlin-1 and Gremlin-2: the murine
parental antibody **Mu-αGREM** (`Mu-aGREM`) and its humanised variant
**Hu-αGREM** (`Hu-aGREM`).

The method has two stages.

**Stage 1** scores every position of the antibody variable domains by combining
a predicted interaction probability, structural proximity to the epitope,
sequence variability across the antibody repertoire, and the rarity of the
wild-type residue, into a single paratoping score `R_i = P_i · D_i · C_i · F_i`.

**Stage 2** ranks all 19 substitutions at each selected position with a
Position-Specific BLOSUM Dissimilarity Matrix (PSBDM), favouring substitutions
that are both chemically dissimilar to the wild type and rare at that position
in the repertoire.

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/AmirAsgary/PaperAnalysis_Collection.git
cd PaperAnalysis_Collection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The input data is distributed separately (≈ 130 MB unpacked) under 
https://doi.org/10.5281/zenodo.22655740. To obtain it, contact 
amir.asgary@mpinat.mpg.de; it becomes publicly available after
publication. Unpack it so that `data/` sits next to `run_analysis.py`.

```
data/
├── antibodies/<antibody>/
│   ├── heavy.fasta, light.fasta      variable-domain sequences
│   ├── proabc2_{heavy,light}.csv     ProABC-2 probabilities (pt, hb, hy)
│   └── msa_{heavy,light}.fa          176,895 paired antibodies from PLAbDab,
│                                     aligned and anchored on this antibody
├── antigens/                         5AEJ, 5HK5 and the trimmed AlphaFold3 models
├── docking/<complex>/
│   └── cluster<NN>_rank<M>.pdb       40 HADDOCK models: the top 10 clusters
│                                     by HADDOCK score × their top 4 members
└── alphafold/
    ├── cluster_predictions/<complex>/
    │   ├── cluster<NN>.pdb           one AlphaFold model per cluster, predicted
    │   ├── cluster<NN>_plddt.npy     from all four members of that cluster
    │   └── cluster<NN>_pae.npy       as templates
    └── relaxed/                      Rosetta-relaxed, for visualisation
```

**Repertoire alignments** are *query-anchored*: the first record is the antibody
itself, it carries no gaps, and the alignment length equals the chain length, so
column *k* is position *k* of the antibody. The pipeline verifies this on load
and aborts if it is violated.

**Structures** carry three chains — `H` (heavy), `L` (light), `G` (antigen) —
each numbered from 1, so residue *i* of chain H is position *i* of the heavy
chain. The epitope is antigen residues 29–45.

The two structure sets play different roles and never mix: the 40 docked models
drive the epitope-contact analysis, while the 10 per-cluster AlphaFold models
define the distance score `D_i`, one per binding orientation so that a heavily
populated cluster cannot dominate.

## Run

```bash
python run_analysis.py
```

One command, no arguments, ~10 minutes on a laptop. It reads `data/`, writes
`results/`, and is deterministic. Two optional flags expose the Stage 2
parameters for sensitivity checks:

```bash
python run_analysis.py --weight 1.5 --pseudocount 0.005
```

## Output

```
results/
├── SUMMARY.txt, summary.json         overview, human- and machine-readable
├── scores/<complex>_<chain>_stage1.csv
│       per position: P_i, D_i, C_i, F_i, R_i, whether selected, rank
├── mutations/<complex>_<chain>_ranked.csv
│       per selected position: wild type, the five most disruptive
│       substitutions and their PSBDM scores
│   ├── ..._PSBDM.csv                 full 20 × N matrix
│   └── ..._msa_logodds.csv           the P(k,j) term alone
├── epitope_contacts/<complex>/       antibody partners of each epitope residue
└── figures/
    ├── epitope_logos/, epitope_profiles/    epitope engagement
    ├── stage1_profiles/              P_i and R_i along each chain
    ├── psbdm_heatmaps/               PSBDM at the top 10 positions
    ├── grem1_vs_grem2/, top_positions/      R_i against GREM1 vs GREM2
    ├── mutation_candidates/          focused heat map of the final candidates
    └── supplementary/                AlphaFold model confidence
```

Every figure is written together with a `.csv` holding exactly the plotted
values, so panels can be re-drawn or checked without rerunning the pipeline.

## How the input data was produced

Recorded for completeness; none of it needs to be repeated.

1. **Antigen structures** — GREM1 from PDB [5AEJ](https://www.rcsb.org/structure/5AEJ),
   GREM2 from [5HK5](https://www.rcsb.org/structure/5HK5); both also predicted
   with [AlphaFold3](https://alphafoldserver.com/), compared against structural
   homologues with [FoldSeek](https://search.foldseek.com/), and trimmed of
   low-confidence regions.
2. **Paratope prediction** — [ProABC-2](https://github.com/haddocking/proABC-2).
3. **Docking** — [HADDOCK 2.4](https://www.bonvinlab.org/software/haddock2.4/),
   active residues from ProABC-2 at P > 0.3, epitope restrained to antigen
   residues 29–45. The server returns the 10 best-scoring clusters with their 4
   best members each.
4. **Structure prediction** — one AlphaFold model per cluster with
   [alphafold_finetune](https://github.com/phbradley/alphafold_finetune), using
   all four members of that cluster as templates. The best model per complex was
   relaxed with Rosetta for visualisation.
5. **Repertoire alignments** — paired sequences from
   [PLAbDab](https://opig.stats.ox.ac.uk/webapps/plabdab/), aligned with
   ClustalOmega, then anchored on each antibody.

`tools/` records how the released tree was derived: `build_dataset.py` for the
docked and relaxed structures, `build_af_dataset.py` for the per-cluster
AlphaFold models, `make_affine_inputs.py` for the AFfine input tables, and
`make_manifest.py` for the dataset checksums. All are provenance documentation
and none needs to be run.

## Citing the tools used

HADDOCK 2.4 (Honorato et al., 2024) · ProABC-2 (Ambrosetti et al., 2020) ·
AlphaFold (Abramson et al., 2024) · alphafold_finetune (Bradley, 2022) ·
FoldSeek (van Kempen et al., 2024) · PLAbDab (Abanades et al., 2024) ·
BLOSUM62 (Henikoff & Henikoff, 1992) · Rosetta · UCSF ChimeraX (Pettersen et al., 2021).

## License

Academic use. Please contact the authors for other licensing.
