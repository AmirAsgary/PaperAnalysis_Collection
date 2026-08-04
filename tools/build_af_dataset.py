#!/usr/bin/env python3
"""
Turn raw AFfine output into ``data/alphafold/cluster_predictions/``.

    python tools/build_af_dataset.py --predictions <affine_output_dir>

One AlphaFold structure was predicted per HADDOCK cluster, using all four
members of that cluster as templates.  AFfine writes each prediction as a single
chain A holding the concatenated target sequence, with AlphaFold's +200
residue-numbering offset between chains.  The analysis expects the docked-model
convention -- chains H, L and G, each numbered from 1 -- so this script splits
and renumbers, and copies the per-residue confidence arrays alongside:

    data/alphafold/cluster_predictions/<complex>/cluster<NN>.pdb
    data/alphafold/cluster_predictions/<complex>/cluster<NN>_plddt.npy
    data/alphafold/cluster_predictions/<complex>/cluster<NN>_pae.npy

Like ``tools/build_dataset.py`` this records how the released tree was derived
and does not need to be run to reproduce the analysis.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Polypeptide import index_to_one, is_aa, three_to_index

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abgrem import config  # noqa: E402

#: Chain ids in the order AFfine concatenates them into the target sequence.
CHAIN_ORDER = ("H", "L", "G")

_PARSER = PDBParser(QUIET=True)


def split_and_renumber(pdb_path: str, lengths: dict[str, int]):
    """Turn a single-chain AFfine prediction into chains H, L, G numbered from 1."""
    structure = _PARSER.get_structure("model", pdb_path)
    model = structure[0]

    residues = [r for chain in model for r in chain if is_aa(r, standard=True)]
    total = sum(lengths[c] for c in CHAIN_ORDER)
    if len(residues) != total:
        raise ValueError(f"{pdb_path}: {len(residues)} residues, expected {total}")

    for chain in list(model):
        model.detach_child(chain.id)

    cursor = 0
    for chain_id in CHAIN_ORDER:
        chain = Chain(chain_id)
        for offset in range(lengths[chain_id]):
            residue = residues[cursor + offset]
            residue.detach_parent()
            residue.id = (" ", offset + 1, " ")
            chain.add(residue)
        model.add(chain)
        cursor += lengths[chain_id]
    return structure


def sequence_of(structure, chain_id: str) -> str:
    return "".join(
        index_to_one(three_to_index(r.resname))
        for r in structure[0][chain_id]
        if is_aa(r, standard=True)
    )


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=os.path.join(root, "data"))
    parser.add_argument("--predictions", required=True,
                        help="directory of raw AFfine output (.pdb and .npy)")
    args = parser.parse_args()

    io = PDBIO()
    written = 0

    for pair in config.COMPLEXES:
        antigen = f"grem{pair.grem}"

        reference_dir = os.path.join(args.data, "docking", pair.name)
        first = sorted(f for f in os.listdir(reference_dir) if f.endswith(".pdb"))[0]
        reference = _PARSER.get_structure("ref", os.path.join(reference_dir, first))
        lengths = {
            "H": len(pair.antibody.heavy),
            "L": len(pair.antibody.light),
            "G": sum(1 for r in reference[0]["G"] if is_aa(r, standard=True)),
        }

        out_dir = os.path.join(args.data, "alphafold", "cluster_predictions", pair.name)
        os.makedirs(out_dir, exist_ok=True)

        prefix = f"cluster_{antigen}_{pair.antibody.name}_"
        found = sorted(f for f in os.listdir(args.predictions)
                       if f.startswith(prefix) and f.endswith(".pdb"))
        if not found:
            print(f"no predictions for {pair.name} (prefix {prefix})", file=sys.stderr)
            return 1

        for filename in found:
            stem = filename[:-len(".pdb")]
            cluster = filename[len(prefix):].split("_model")[0]

            structure = split_and_renumber(
                os.path.join(args.predictions, filename), lengths
            )
            if sequence_of(structure, "H") != pair.antibody.heavy:
                raise ValueError(f"{filename}: chain H does not match config heavy")
            if sequence_of(structure, "L") != pair.antibody.light:
                raise ValueError(f"{filename}: chain L does not match config light")
            epitope = "".join(
                sequence_of(structure, "G")[i - 1] for i in config.EPITOPE_RESIDUE_IDS
            )
            if epitope != pair.epitope:
                raise ValueError(
                    f"{filename}: epitope 29-45 is {epitope}, expected {pair.epitope}"
                )

            io.set_structure(structure)
            io.save(os.path.join(out_dir, f"{cluster}.pdb"))

            for source_suffix, target_suffix in (
                ("_plddt.npy", "_plddt.npy"),
                ("_predicted_aligned_error.npy", "_pae.npy"),
            ):
                source = os.path.join(args.predictions, stem + source_suffix)
                if not os.path.exists(source):
                    raise FileNotFoundError(
                        f"{source} is missing; pLDDT and PAE are required for the "
                        f"confidence figure"
                    )
                shutil.copyfile(source, os.path.join(out_dir, cluster + target_suffix))
            written += 1

        print(f"{pair.label}: {len(found)} structures -> {out_dir}")

    print(f"\n{written} predictions written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
