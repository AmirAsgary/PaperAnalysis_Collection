# Paratope scoring and mutation prioritisation for anti-GREM antibodies

Code and data to reproduce every figure and table of the manuscript.

Two antibodies are analysed against both Gremlin-1 and Gremlin-2:

| Antibody | Directory name | Description |
|---|---|---|
| Mu-αGREM | `Mu-aGREM` | murine parental antibody |
| Hu-αGREM | `Hu-aGREM` | humanised variant |

The method has two stages. **Stage 1** scores every position of the antibody
variable domains by combining a predicted interaction probability, structural
proximity to the epitope in docked models, sequence variability across the
antibody repertoire, and the rarity of the wild-type residue, into a single
rank `R_i = P_i · D_i · C_i · F_i`. **Stage 2** ranks all 19 substitutions at
each selected position with a Position-Specific BLOSUM Dissimilarity Matrix
(PSBDM), which favours substitutions that are both chemically dissimilar to the
wild type and rare at that position in the repertoire.

---

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/AmirAsgary/PaperAnalysis_Collection.git
cd PaperAnalysis_Collection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The input data is distributed separately (≈ 130 MB unpacked). Download it into
the repository root so that `data/` sits next to `run_analysis.py`:

```bash
wget -O data.zip https://owncloud.gwdg.de/index.php/s/FI9vn0vDYhn9UZD/download
unzip data.zip
```

## Run

```bash
python run_analysis.py
```

One command, no arguments, ~10 minutes on a laptop. It reads `data/`, writes
`results/`, and is deterministic — repeating it reproduces byte-identical
tables.

Two optional flags expose the Stage 2 parameters, for sensitivity checks:

```bash
python run_analysis.py --weight 1.5 --pseudocount 0.005
```

---

## What is in `data/`

```
data/
├── antibodies/<antibody>/
│   ├── heavy.fasta, light.fasta      variable-domain sequences
│   ├── proabc2_heavy.csv             ProABC-2 per-residue interaction
│   ├── proabc2_light.csv             probabilities (pt, hb, hy)
│   ├── msa_heavy.fa                  176,895 paired antibodies from PLAbDab,
│   └── msa_light.fa                  aligned and anchored on this antibody
│
├── antigens/
│   ├── GREM1_5AEJ.pdb                crystal structures from the PDB
│   ├── GREM2_5HK5.pdb
│   ├── GREM1_alphafold3_trimmed.cif  AlphaFold3 models after removal of
│   └── GREM2_alphafold3_trimmed.cif  low-confidence regions
│
├── docking/<antibody>_GREM<n>/
│   └── cluster<NN>_rank<M>.pdb       40 HADDOCK models: the top 10 clusters
│                                     by HADDOCK score × their top 4 members
│
└── alphafold/
    ├── final_models/                 the four complexes used in the manuscript
    ├── relaxed/                      Rosetta-relaxed, for visualisation
    └── template_benchmark/           per-run pLDDT and PAE tables
```

**Repertoire alignments** (`msa_*.fa`) are *query-anchored*: the first record is
the antibody itself, it carries no gaps, and the alignment length equals the
chain length. Column *k* is therefore position *k* of the antibody, with no
index mapping. The pipeline verifies this on load and aborts if it is violated.

**Docked models** carry three chains — `H` (heavy), `L` (light), `G` (antigen) —
each numbered from 1, so residue *i* of chain H is position *i* of the heavy
chain sequence, directly comparable to the score tables. The epitope is antigen
residues 29–45. The `_rank1` model of each cluster is the cluster
representative used for the distance score `D_i`; all 40 models are used for the
epitope-contact analysis.

`tools/build_dataset.py` records how this tree was derived from the raw HADDOCK,
ProABC-2 and AlphaFold output. It is provenance documentation and does not need
to be run.

---

## What you get in `results/`

```
results/
├── SUMMARY.txt                       human-readable overview
├── summary.json                      the same, machine-readable
│
├── scores/<complex>_<chain>_stage1.csv
│       per position: P_i, D_i, C_i, F_i, R_i, whether it was selected, rank
│
├── mutations/<complex>_<chain>_ranked.csv
│       per selected position: wild type, the five most disruptive
│       substitutions and their PSBDM scores
│   ├── ..._PSBDM.csv                 full 20 × N matrix
│   └── ..._msa_logodds.csv           the P(k,j) term alone
│
├── epitope_contacts/<complex>/
│   ├── <residue><n>_partners.csv     antibody residues contacting each
│   │                                 epitope position, and in how many models
│   └── epitope_summary.csv           per-position contact totals, heavy/light split
│
└── figures/
    ├── epitope_logos/                sequence logo of epitope engagement
    ├── epitope_profiles/             the same as a line profile
    ├── stage1_profiles/              P_i and R_i along each chain
    ├── psbdm_heatmaps/               PSBDM scores at the top 10 positions
    ├── grem1_vs_grem2/               R_i against GREM1 vs GREM2
    ├── top_positions/                the 10 highest-scoring positions
    ├── mutation_candidates/          focused heat map of the final candidates
    └── supplementary/                supporting evidence for methodological
                                      choices, not main-text panels
```

Every figure is written together with a `.csv` holding exactly the plotted
values, so panels can be re-drawn or checked without rerunning the pipeline.

---

## How the input data was produced

These steps are recorded for completeness; their outputs are in `data/` and none
of them needs to be repeated to reproduce the results.

1. **Antigen structures** — GREM1 from PDB [5AEJ](https://www.rcsb.org/structure/5AEJ),
   GREM2 from [5HK5](https://www.rcsb.org/structure/5HK5); both also predicted
   with [AlphaFold3](https://alphafoldserver.com/), compared against structural
   homologues with [FoldSeek](https://search.foldseek.com/), and trimmed of
   low-confidence regions.
2. **Paratope prediction** — [ProABC-2](https://github.com/haddocking/proABC-2)
   on each antibody, giving `proabc2_*.csv`.
3. **Docking** — [HADDOCK 2.4](https://www.bonvinlab.org/software/haddock2.4/),
   active residues from ProABC-2 at P > 0.3, epitope restrained to antigen
   residues 29–45 plus their most solvent-exposed nearest neighbours. The server
   returns the 10 best-scoring clusters with their 4 best members each.
4. **Structure prediction** — the docked models were used as AlphaFold
   templates, one template per prediction, and the highest-pLDDT model of each
   complex was retained (`data/alphafold/final_models/`, pLDDT 96.1–97.2).
   Supplying several docked poses at once was also tested and scored far worse;
   `run_analysis.py` reproduces that comparison as a supplementary figure.
5. **Repertoire alignments** — paired sequences from
   [PLAbDab](https://opig.stats.ox.ac.uk/webapps/plabdab/), aligned with
   ClustalOmega, then anchored on each antibody by deleting alignment columns
   that are gaps in that antibody.

## Citing the tools used

HADDOCK 2.4 (Honorato et al., 2024) · ProABC-2 (Ambrosetti et al., 2020) ·
AlphaFold (Abramson et al., 2024) · FoldSeek (van Kempen et al., 2024) ·
PLAbDab (Abanades et al., 2024) · BLOSUM62 (Henikoff & Henikoff, 1992) ·
Rosetta · UCSF ChimeraX (Pettersen et al., 2021).

## License

Academic use. Please contact the authors for other licensing.
