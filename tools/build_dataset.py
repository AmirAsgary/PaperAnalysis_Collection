#!/usr/bin/env python3
"""
Build the released `data/` tree from the original working directory.
=====================================================================

This script is provenance documentation: it records exactly which files in the
original (messy) analysis directory became which files in the released dataset,
and what transformation was applied.  It is NOT part of the analysis pipeline
and does not need to be run by anyone reproducing the paper — the released
`data/` archive is its output.

Transformations applied
-----------------------
1.  Antibody aliases           14D10    -> Mu-aGREM   (murine anti-GREM)
                               Hu-Var-7 -> Hu-aGREM   (humanised anti-GREM)
    The 3A1-3 antibody is dropped entirely (not part of the manuscript).

2.  HADDOCK models.  The original directory carried each docked model twice:
    once as raw server output (`cluster{N}_{M}.pdb`, antibody heavy+light fused
    into chain B, antigen in chain A) and once chain-split (`modified_*.pdb`,
    chains H/L/G but residues numbered continuously 1..328 across chains).
    A third copy of the 10 cluster representatives lived under
    `alphafold/templates/` (chain-split AND renumbered per chain).

    The release keeps ONE copy of each model, chain-split (H/L/G) and renumbered
    per chain from 1, so that residue numbers in every output file line up with
    the antibody sequence position used by the scoring code.

3.  File names are made self-describing:
        cluster{N}_{M}.pdb  ->  cluster{N:02d}_rank{M}.pdb

Usage
-----
    python tools/build_dataset.py --source <original-dir> --dest data
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

import pandas as pd
from Bio import PDB, SeqIO

# ─────────────────────────────────────────────────────────────────────────────
# Mapping from the original working directory to the released layout
# ─────────────────────────────────────────────────────────────────────────────
ANTIBODIES = {
    "Hu-aGREM": {
        "legacy_dir": "Hu-Var-7",
        "legacy_msa": "huvar7",
        "legacy_af": "huvar7",
        "haddock": {1: "Hu-Var-7/395611-HuVar_GREM1_summary",
                    2: "Hu-Var-7/395612-HuVar_GREM2_summary"},
    },
    "Mu-aGREM": {
        "legacy_dir": "14D10",
        "legacy_msa": "14d10",
        "legacy_af": "14d10",
        "haddock": {1: "14D10/395665-GREM1_14D10_summary",
                    2: "14D10/395672-GREM2_14D10_summary"},
    },
}

# The docked complexes carry the antibody in chain B and the antigen in chain A.
AB_CHAIN, ANTIGEN_CHAIN = "B", "A"

ANTIGENS = {
    "GREM1_5AEJ.pdb": "pdb/GREM1_monomer.pdb",
    "GREM2_5HK5.pdb": "pdb/GREM2.pdb",
    "GREM1_alphafold3_trimmed.cif": "pdb/GREM1_modified.cif",
    "GREM2_alphafold3_trimmed.cif": "pdb/GREM2_modified.cif",
}

AF_RELAXED = {
    "Hu-aGREM_GREM1_relaxed.pdb": "alphafold/relaxed/GREM1_Huvar7.pdb",
    "Hu-aGREM_GREM2_relaxed.pdb": "alphafold/relaxed/grem2_huvar7.pdb",
}

CLUSTER_RE = re.compile(r"^cluster(\d+)_(\d+)\.pdb$")


# ─────────────────────────────────────────────────────────────────────────────
# Structure preparation
# ─────────────────────────────────────────────────────────────────────────────
def split_and_renumber(src_pdb: str, dest_pdb: str, heavy_len: int) -> None:
    """
    Convert one raw HADDOCK model into the released form.

    The raw model has the antibody heavy and light chains fused into a single
    chain (`AB_CHAIN`) and the antigen in `ANTIGEN_CHAIN`.  The first
    `heavy_len` residues of the fused chain are the heavy chain; the remainder
    is the light chain.  Output chains are H (heavy), L (light) and G (antigen),
    each renumbered from 1 so that residue *i* of chain H is position *i* of the
    heavy-chain sequence.
    """
    structure = PDB.PDBParser(QUIET=True).get_structure("m", src_pdb)
    model = structure[0]

    fused = model[AB_CHAIN]
    antigen = model[ANTIGEN_CHAIN]

    heavy = PDB.Chain.Chain("H")
    light = PDB.Chain.Chain("L")
    for index, residue in enumerate(list(fused)):
        (heavy if index < heavy_len else light).add(residue)

    out_model = PDB.Model.Model(0)
    out_structure = PDB.Structure.Structure("released")
    out_structure.add(out_model)

    # Chain order H, L, G keeps the antibody first, matching the score tables.
    for chain in (heavy, light, antigen):
        released = PDB.Chain.Chain("G" if chain is antigen else chain.id)
        for number, residue in enumerate(list(chain), start=1):
            new_residue = PDB.Residue.Residue(
                (" ", number, " "), residue.get_resname(), residue.get_segid()
            )
            for atom in residue:
                new_residue.add(atom.copy())
            released.add(new_residue)
        out_model.add(released)

    io = PDB.PDBIO()
    io.set_structure(out_structure)
    io.save(dest_pdb)


def first_sequence(fasta_path: str) -> str:
    return str(next(SeqIO.parse(fasta_path, "fasta")).seq)


# ─────────────────────────────────────────────────────────────────────────────
# Build steps
# ─────────────────────────────────────────────────────────────────────────────
def build_antibodies(source: str, dest: str) -> dict[str, int]:
    heavy_lengths = {}
    for name, meta in ANTIBODIES.items():
        legacy = os.path.join(source, meta["legacy_dir"])
        msa = os.path.join(source, "antibody_sequences", meta["legacy_msa"])
        out = os.path.join(dest, "antibodies", name)
        os.makedirs(out, exist_ok=True)

        shutil.copy2(os.path.join(legacy, "heavy.fasta"), os.path.join(out, "heavy.fasta"))
        shutil.copy2(os.path.join(legacy, "light.fasta"), os.path.join(out, "light.fasta"))
        shutil.copy2(os.path.join(legacy, "heavy-pred.csv"), os.path.join(out, "proabc2_heavy.csv"))
        shutil.copy2(os.path.join(legacy, "light-pred.csv"), os.path.join(out, "proabc2_light.csv"))
        shutil.copy2(os.path.join(msa, f'aln_heavy_{meta["legacy_msa"]}.fa'),
                     os.path.join(out, "msa_heavy.fa"))
        shutil.copy2(os.path.join(msa, f'aln_light_{meta["legacy_msa"]}.fa'),
                     os.path.join(out, "msa_light.fa"))

        heavy_lengths[name] = len(first_sequence(os.path.join(out, "heavy.fasta")))
        print(f"  antibodies/{name}: 6 files (heavy chain {heavy_lengths[name]} aa)")
    return heavy_lengths


def build_antigens(source: str, dest: str) -> None:
    out = os.path.join(dest, "antigens")
    os.makedirs(out, exist_ok=True)
    for released_name, legacy_path in ANTIGENS.items():
        shutil.copy2(os.path.join(source, legacy_path), os.path.join(out, released_name))
    print(f"  antigens/: {len(ANTIGENS)} files")


def build_docking(source: str, dest: str, heavy_lengths: dict[str, int]) -> None:
    for name, meta in ANTIBODIES.items():
        for grem, legacy_dir in meta["haddock"].items():
            pair = f"{name}_GREM{grem}"
            out = os.path.join(dest, "docking", pair)
            os.makedirs(out, exist_ok=True)

            src_dir = os.path.join(source, legacy_dir)
            raw = sorted(f for f in os.listdir(src_dir) if CLUSTER_RE.match(f))
            for filename in raw:
                cluster, rank = (int(g) for g in CLUSTER_RE.match(filename).groups())
                split_and_renumber(
                    os.path.join(src_dir, filename),
                    os.path.join(out, f"cluster{cluster:02d}_rank{rank}.pdb"),
                    heavy_lengths[name],
                )
            clusters = sorted({int(CLUSTER_RE.match(f).group(1)) for f in raw})
            print(f"  docking/{pair}: {len(raw)} models, "
                  f"{len(clusters)} clusters {clusters}")


def build_alphafold(source: str, dest: str) -> None:
    """
    Copy the Rosetta-relaxed structures used for visualisation.

    The per-cluster AlphaFold models that define D_i are built separately, by
    ``tools/build_af_dataset.py``, from raw AFfine output.
    """
    out_relaxed = os.path.join(dest, "alphafold", "relaxed")
    os.makedirs(out_relaxed, exist_ok=True)
    for released_name, legacy_path in AF_RELAXED.items():
        shutil.copy2(os.path.join(source, legacy_path), os.path.join(out_relaxed, released_name))
    print(f"  alphafold/relaxed: {len(AF_RELAXED)} structures")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="original analysis directory")
    parser.add_argument("--dest", default="data", help="output data directory")
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"source directory not found: {args.source}", file=sys.stderr)
        return 1

    os.makedirs(args.dest, exist_ok=True)
    print(f"Building dataset in {args.dest}/ from {args.source}")
    heavy_lengths = build_antibodies(args.source, args.dest)
    build_antigens(args.source, args.dest)
    build_docking(args.source, args.dest, heavy_lengths)
    build_alphafold(args.source, args.dest)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
