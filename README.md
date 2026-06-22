# Antibody–GREM Paratope Scoring and Mutation Suggestion

A two-stage computational pipeline for identifying candidate mutation positions on antibodies targeting GREM1/GREM2 and suggesting amino-acid substitutions using structural, conservation, and interaction data.

## Overview

Three antibodies are analysed against both GREM1 and GREM2:

| Antibody | Alias | GREM targets |
|----------|-------|-------------|
| Hu-Var-7 | huvar7 | GREM1, GREM2 |
| 3A1-3 | 3a13 | GREM1, GREM2 |
| 14D10 | 14d10 | GREM1, GREM2 |

The pipeline uses a two-stage approach:

**Stage 1 — Position Selection.**
For each residue position *i* on the antibody heavy/light chain:
1. Compute P_i, the interaction probability from ProABC-2 (union of hydrophobic, general, and hydrogen-bond probabilities).
2. Compute D_i, the distance score from docked structures (how close position *i* is to the epitope).
3. Compute C_i, the conservation score from antibody MSAs (based on Shannon entropy — high variability across antibodies means the position is in a hypervariable region and suitable for mutation).
4. Compute F_i = exp(−15 × Freq_wt_i), which penalises positions where the wild-type amino acid is very common in the antibody repertoire (frequency > ~10% → F_i ≈ 0).
5. Filter: keep only positions where P_i > 0.5.
6. Rank selected positions by R_i = P_i × D_i × C_i × F_i.

**Stage 2 — Mutation Ranking via Position-Specific BLOSUM Dissimilarity Matrix (PSBDM).**
For each selected position *k* with wild-type amino acid *i*, rank all 19 possible substitutions *j* using a combined score that integrates BLOSUM62 similarity with position-specific amino acid frequencies from the MSA:

```
Score(j) = B(i, j) + w × P(k, j)
```

Where:
- **B(i, j)** = BLOSUM62 score between wild-type *i* and mutant *j*
- **w** = weight parameter (default = 1.0) to balance BLOSUM vs. MSA contributions
- **P(k, j)** = position-specific MSA score = 2 × log₂((f_k(j) + ε) / q_j)
  - f_k(j) = observed frequency of amino acid *j* at position *k* in the MSA
  - q_j = background frequency of amino acid *j* (BLOSUM62 background frequencies)
  - ε = pseudocount (default = 0.01) to avoid log(0)

**Lower scores indicate better mutations** — amino acids that are both chemically dissimilar (low BLOSUM score) and rare at that specific position in the antibody repertoire.

## Prerequisites

### Software 
Used for research, not required for replicating the results

- Python ≥ 3.8
- [HADDOCK 2.4](https://www.bonvinlab.org/software/haddock2.4/) — antibody–antigen docking
- [ProABC-2](https://github.com/haddocking/proABC-2) — paratope prediction
- [AlphaFold](https://github.com/google-deepmind/alphafold) (or AlphaFold3 Server) — structure prediction
- [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/) — visualisation (optional)
- [ClustalOmega](http://www.clustal.org/omega/) — MSA computation
- [Rosetta](https://www.rosettacommons.org/) — structure relaxation (optional)

### Python packages

```bash
pip install numpy pandas matplotlib seaborn biopython scipy logomaker
```

### Data
```bash
git clone https://github.com/AmirAsgary/PaperAnalysis_Collection.git
cd PaperAnalysis_Collection/
wget -O data.zip https://owncloud.gwdg.de/index.php/s/FI9vn0vDYhn9UZD/download
unzip data.zip
```
## Repository Structure

```
.
├── run_analysis.py                  # Main pipeline script
├── scripts.py                       # Utility functions (distance, conservation, BLOSUM, PSBDM, PDB handling)
├── README.md
├── .gitignore
│
├── 14D10/                           # Antibody-specific data
│   ├── heavy.fasta                  # Heavy-chain FASTA
│   ├── light.fasta                  # Light-chain FASTA
│   ├── 14D10-features.csv           # ProABC-2 features
│   ├── heavy-pred.csv               # ProABC-2 interaction probabilities (heavy)
│   ├── light-pred.csv               # ProABC-2 interaction probabilities (light)
│   ├── 395665-GREM1_14D10_summary/  # HADDOCK docking output (GREM1)
│   │   ├── cluster*_*.pdb           # Top-4 docked models per cluster
│   │   ├── modified_cluster*_*.pdb  # Chain-split and renumbered PDBs
│   │   └── csvs_grem1/              # Epitope interaction CSVs
│   └── 395672-GREM2_14D10_summary/  # HADDOCK docking output (GREM2)
│
├── 3A1-3/                           # Same structure as 14D10/
│   ├── heavy-pred.csv
│   ├── light-pred.csv
│   └── ...
│
├── Hu-Var-7/                        # Same structure as 14D10/
│   ├── heavy-pred.csv
│   ├── light-pred.csv
│   └── ...
│
├── antibody_sequences/              # MSA data for conservation analysis
│   ├── huvar7/
│   │   ├── aln_heavy_huvar7.fa      # ClustalOmega MSA (heavy)
│   │   └── aln_light_huvar7.fa      # ClustalOmega MSA (light)
│   ├── 3a13/
│   │   ├── aln_heavy_3a13.fa
│   │   └── aln_light_3a13.fa
│   └── 14d10/
│       ├── aln_heavy_14d10.fa
│       └── aln_light_14d10.fa
│
├── alphafold/templates/             # AlphaFold-predicted structures (used as docked PDBs)
│   ├── huvar7_grem1/                # PDB files for Hu-Var-7 × GREM1
│   ├── huvar7_grem2/
│   ├── 3a13_grem1/
│   ├── 3a13_grem2/
│   ├── 14d10_grem1/
│   └── 14d10_grem2/
│
├── pdb/                             # Reference GREM structures
│   ├── GREM1_monomer.pdb            # PDB: 5AEJ
│   └── GREM2.pdb                    # PDB: 5HK5
│
├── seqlogo/                         # Epitope logo plots (generated)
│   ├── 14D10 GEREM1.png
│   ├── Hu-Var-7 GEREM2.pdf
│   └── ...
│
└── results/                         # Pipeline output (generated)
    ├── SUMMARY_REPORT.txt
    ├── summary.json
    ├── figures/
    │   ├── huvar7_grem_heavy.png
    │   ├── huvar7_grem_heavy_data.csv      # Data for each figure
    │   ├── huvar7_grem_light.png
    │   ├── huvar7_mutation_heatmap_data.csv
    │   └── ...
    ├── huvar7_matrices/             # Full PSBDM and P(k,j) matrices for HuVar7
    │   ├── huvar7_heavy_PSBDM_full.csv
    │   ├── huvar7_heavy_Pkj_raw.csv
    │   ├── huvar7_light_PSBDM_full.csv
    │   └── huvar7_light_Pkj_raw.csv
    ├── huvar7_grem1/
    │   ├── huvar7_grem1_heavychain_stage1.csv
    │   ├── huvar7_grem1_heavychain_PSBDM.csv    # Position-Specific BLOSUM Dissimilarity Matrix
    │   ├── huvar7_grem1_heavychain_Pkj.csv      # Raw P(k,j) values from MSA
    │   ├── huvar7_grem1_heavychain_mutations.csv
    │   ├── huvar7_grem1_heavy_profile.png
    │   ├── huvar7_grem1_heavy_profile_data.csv
    │   ├── huvar7_grem1_heavy_psbdm_heatmap.png
    │   └── ... (same for light)
    ├── huvar7_grem1_heavy_Ri_scores.csv         # R_i scores for all positions
    ├── huvar7_grem1_light_Ri_scores.csv
    └── ... (one folder per Ab-GREM pair)
```

## How to Reproduce the Full Analysis

### Step 1 — Obtain Antigen Structures

Download GREM1 (PDB: [5AEJ](https://www.rcsb.org/structure/5AEJ)) and GREM2 (PDB: [5HK5](https://www.rcsb.org/structure/5HK5)) from the Protein Data Bank. Place them in `pdb/`. Additionally, predict both structures using AlphaFold3 and select the highest-confidence models.

### Step 2 — Predict Antibody Structures

Predict each antibody structure using AlphaFold3. Select the model with the highest pLDDT/ipTM. Remove low-confidence disordered regions.

### Step 3 — Predict Paratope Residues (ProABC-2)

Install ProABC-2 and run it on each antibody:

```bash
git clone https://github.com/haddocking/proABC-2.git
cd proABC-2
# Follow installation instructions in the ProABC-2 README

# For each antibody, generate heavy-pred.csv and light-pred.csv
# Place them in the corresponding antibody folder (e.g., 14D10/)
```

The output CSV files contain per-residue probabilities for general (pt), hydrogen-bond (hb), and hydrophobic (hy) interactions.

### Step 4 — Dock Antibody–Antigen Complexes (HADDOCK)

Run HADDOCK 2.4 for each antibody–GREM pair:

1. Define active residues from ProABC-2 predictions (P > 0.3).
2. For the epitope, use experimentally defined residues (GREM residues 29–45) plus their nearest Euclidean neighbour with highest SASA.
3. Run two docking experiments per pair: (a) without epitope specification, (b) with defined epitope.
4. Select the top-4 models from each HADDOCK cluster based on HADDOCK score.

Output PDBs go into the antibody-specific summary folders (e.g., `14D10/395665-GREM1_14D10_summary/`).

### Step 5 — Split and Renumber Docked PDBs

The docked PDBs have antibody heavy+light chains fused into one chain. Split them:

```python
import scripts
scripts.split_chain_and_rename(
    'path/to/summary_folder',
    'path/to/heavy.fasta',
    'path/to/light.fasta',
    AB='B',    # antibody chain ID in HADDOCK output
    Grem='A'   # GREM chain ID
)
```

This produces `modified_*.pdb` files with chains H (heavy), L (light), and G (GREM).

### Step 6 — Predict Final Structures with AlphaFold (Template-Guided)

Use the top-4 HADDOCK docked models as templates for AlphaFold:

1. For each HADDOCK cluster, run AlphaFold with the docked structure as a template.
2. Select the best prediction per cluster, then combine all cluster-best models as templates for a final prediction.
3. Optionally relax with Rosetta.
4. Place the final PDB files in `alphafold/templates/<ab>_<grem>/`.

### Step 7 — Prepare Antibody MSAs for Conservation Scoring

Download paired antibody sequences from [PLAbDab](https://opig.stats.ox.ac.uk/webapps/plabdab/).

```bash
# Align with ClustalOmega (install: http://www.clustal.org/omega/)
clustalo -i antibody_sequences/huvar7/heavy_seqs.fa \
         -o antibody_sequences/huvar7/aln_heavy_huvar7.fa \
         --outfmt=fasta --threads=4

clustalo -i antibody_sequences/huvar7/light_seqs.fa \
         -o antibody_sequences/huvar7/aln_light_huvar7.fa \
         --outfmt=fasta --threads=4
```

Repeat for all three antibodies.

### Step 8 — Run the Scoring Pipeline

```bash
# Standard run: scoring + logo plots + all figures
python run_analysis.py

# With PDB cleaning (re-split HADDOCK chains, recompute structural proximity CSVs):
python run_analysis.py --clean

# Customize PSBDM parameters:
python run_analysis.py --weight 1.5 --pseudocount 0.005
```

The `--clean` flag is only needed once (or when HADDOCK outputs change). It:
1. Splits HADDOCK-docked PDBs into H (heavy), L (light), G (GREM) chains.
2. Computes per-epitope residue interaction counts within 5.5 Å.
3. Saves per-epitope CSVs and scatter plots to each HADDOCK summary folder.

Without `--clean`, the pipeline:
1. Generates epitope sequence logo plots from existing CSVs (saved to `seqlogo/`).
2. Runs Stage 1 (position selection) and Stage 2 (mutation ranking via PSBDM) for all 6 Ab–GREM pairs.
3. Saves per-position scores, PSBDM matrices, P(k,j) matrices, and mutation summaries to `results/<ab>/`.
4. Generates GREM1-vs-GREM2 comparison plots in `results/figures/`.
5. Generates HuVar7-specific mutation heatmaps and paratoping bar plots.
6. Saves R_i scores for all positions to CSV files.
7. Saves full PSBDM and P(k,j) matrices for HuVar7.
8. Writes a global summary to `results/SUMMARY_REPORT.txt`.

## Output Files

For each antibody–GREM pair (e.g., `results/huvar7_grem1/`):

| File | Description |
|------|-------------|
| `*_stage1.csv` | Per-position P_i, D_i, C_i, F_i, R_i, and selection flag |
| `*_PSBDM.csv` | Position-Specific BLOSUM Dissimilarity Matrix (20 AA × N positions) |
| `*_Pkj.csv` | Raw P(k,j) scores from MSA (20 AA × N positions) |
| `*_mutations.csv` | Top mutations for each selected position with PSBDM scores |
| `*_profile.png` | P_i and R_i bar profiles along the chain |
| `*_profile_data.csv` | Data for the profile plot |
| `*_psbdm_heatmap.png` | Heatmap of PSBDM scores for top positions |
| `*_psbdm_heatmap_data.csv` | Data for the heatmap |

In `results/`:

| File | Description |
|------|-------------|
| `<ab>_<grem>_heavy_Ri_scores.csv` | R_i scores for all heavy chain positions |
| `<ab>_<grem>_light_Ri_scores.csv` | R_i scores for all light chain positions |

In `results/huvar7_matrices/`:

| File | Description |
|------|-------------|
| `huvar7_heavy_PSBDM_full.csv` | Full PSBDM matrix for HuVar7 heavy chain (20 AA rows × positions) |
| `huvar7_heavy_Pkj_raw.csv` | Raw P(k,j) values for HuVar7 heavy chain |
| `huvar7_light_PSBDM_full.csv` | Full PSBDM matrix for HuVar7 light chain |
| `huvar7_light_Pkj_raw.csv` | Raw P(k,j) values for HuVar7 light chain |

In `results/figures/`:

| File | Description |
|------|-------------|
| `<ab>_grem_heavy.png` | GREM1 vs GREM2 R_i bar plot (heavy chain) |
| `<ab>_grem_heavy_data.csv` | Data for the bar plot |
| `<ab>_grem_light.png` | GREM1 vs GREM2 R_i bar plot (light chain) |
| `<ab>_grem_*_scatter.png` | GREM1 vs GREM2 R_i line overlay |
| `<ab>_grem_*_scatter_data.csv` | Data for the scatter plot |
| `<ab>_paratoping_heavy.png` | Paratoping scores bar plot, black/white (top 10) |
| `<ab>_paratoping_heavy_data.csv` | Data for the paratoping plot |
| `huvar7_mutation_heatmap_*.png` | Mutation heatmaps (green-red and blue-red) |
| `huvar7_mutation_heatmap_data.csv` | Data for the heatmap |

In `seqlogo/`:

| File | Description |
|------|-------------|
| `<Ab> GEREM<N>.png/pdf` | Epitope binding sequence logo per Ab–GREM pair |

In each HADDOCK summary folder (generated by `--clean`):

| File | Description |
|------|-------------|
| `csvs_grem<N>/*.csv` | Per-epitope residue interaction counts and probabilities |
| `csvs_grem<N>/epitope_binding_probablity.csv` | Aggregated epitope binding data |
| `csvs_grem<N>/H_L_prob.csv` | Heavy/light chain interaction breakdown |
| `plots/*.png` | Per-epitope scatter plots and binding probability plots |

## Methodology

See the accompanying manuscript for full details. Briefly:

### Stage 1: Position Selection

- **P_i**: Union probability of interaction from ProABC-2: `P_i = 1 - (1-P_pt)(1-P_hb)(1-P_hy)`. Positions with P_i > 0.5 are considered interacting with the epitope.
- **D_i**: Distance score from docked structures. For each structure, a step function counts how many distance thresholds (4, 5, 6, 7, 8, 12 Å) each residue pair falls within, averaged over thresholds and passed through a sigmoid. Final D_i is the mean over all structures.
- **C_i**: Conservation score from antibody MSAs. Computed as `C_i = σ(2.5 · (-log(Conservation_i + ε) - 1))`, where `Conservation_i = 1 - H_i / H_max`, H_i is the Shannon entropy at position i, and H_max ≈ 4.32. High C_i means high variability (hypervariable region), low C_i means the position is conserved and should not be mutated.
- **F_i**: Wild-type frequency penalty: `F_i = exp(-15 · Freq_wt_i)`, where Freq_wt_i is the frequency of the wild-type amino acid at position i in the antibody MSA. Positions where the wild-type residue is ubiquitous (freq > ~10%) get F_i ≈ 0 and are effectively excluded.
- **R_i = P_i × D_i × C_i × F_i**: The combined position score. Positions that have high interaction probability, high structural proximity to the epitope, high variability, and a rare wild-type residue are ranked highest.

### Stage 2: Mutation Ranking via PSBDM

For each selected position *k* with wild-type amino acid *i*, we compute the **Position-Specific BLOSUM Dissimilarity Matrix (PSBDM)** score for each possible mutant amino acid *j*:

```
Score(j) = B(i, j) + w × P(k, j)
```

Where:
- **B(i, j)** = BLOSUM62 score between wild-type *i* and mutant *j*. Lower (more negative) values indicate greater chemical dissimilarity.
- **w** = weight parameter (default = 1.0) that balances the BLOSUM and MSA contributions.
- **P(k, j)** = position-specific MSA score, calculated as:
  ```
  P(k, j) = 2 × log₂( (f_k(j) + ε) / q_j )
  ```
  - **f_k(j)** = observed frequency of amino acid *j* at position *k* in the antibody MSA
  - **q_j** = uniform background frequency (0.05 = 1/20) to ensure independence between the MSA-derived term and the BLOSUM62 score
  - **ε** = pseudocount (default = 0.01) to avoid log(0) for unobserved amino acids

**Interpretation:**
- **Lower PSBDM scores indicate better mutations** — amino acids that are:
  1. Chemically dissimilar from the wild-type (low BLOSUM score)
  2. Rare at that specific position in the antibody repertoire (negative P(k,j))

The rationale is that such mutations are more likely to significantly alter the antibody's binding properties while still being structurally compatible with the antibody framework.

**Cysteine exclusion:** In cases where cysteine was identified as the top-ranked mutation, the next-best candidate was selected to avoid potential disulfide bond formation that could interfere with antibody production and folding.

## Citation

If you use this pipeline, please cite the relevant tools:
- HADDOCK 2.4: Honorato et al. (2024)
- ProABC-2: Ambrosetti et al. (2020)
- AlphaFold: Abramson et al. (2024)
- PLAbDab: Abanades et al.
- BLOSUM62: Henikoff & Henikoff (1992)

## License

This project is provided for academic use. Please contact the authors for licensing.
