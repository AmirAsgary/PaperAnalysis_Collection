#!/usr/bin/env python3
"""
Build AlphaFold-finetune (AFfine) inputs: one prediction per HADDOCK cluster,
using all four members of that cluster as templates.

    python tools/make_affine_inputs.py

Writes a self-contained bundle to ``affine_input/``:

    targets.tsv                              40 rows, one per cluster
    alignments/aln_<ab>_<ag>_cluster<NN>.tsv  40 files, 4 template rows each
    templates/<complex>/cluster<NN>_rank<M>.pdb   copies of the docked models

The target sequence of every complex is the concatenation heavy/light/antigen,
which is exactly the residue order of the docked PDBs (chains H, L, G, each
numbered from 1).  Target and template are therefore the same sequence and the
alignment is the identity map 0:0;1:1;...;N-1:N-1.  The script verifies this
against the actual PDB contents and aborts if any model disagrees.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import index_to_one, is_aa, three_to_index

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abgrem import config  # noqa: E402

#: Chain order as it appears in the docked PDB files; also the order in which
#: the chains are concatenated into ``target_chainseq``.
CHAIN_ORDER = ("H", "L", "G")

ALIGN_COLUMNS = (
    "template_pdbfile",
    "target_to_template_alignstring",
    "identities",
    "target_len",
    "template_len",
    "use_template",
)
TARGET_COLUMNS = ("targetid", "target_chainseq", "templates_alignfile")

_PARSER = PDBParser(QUIET=True)


def chain_sequences(pdb_path: str) -> dict[str, str]:
    """One-letter sequence of every chain, in file order, checked for 1..N numbering."""
    structure = _PARSER.get_structure("model", pdb_path)
    sequences = {}
    for chain in structure[0]:
        letters, numbers = [], []
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            letters.append(index_to_one(three_to_index(residue.resname)))
            numbers.append(residue.id[1])
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(
                f"{pdb_path}: chain {chain.id} is not numbered 1..N "
                f"({numbers[:3]}...{numbers[-3:]}); the identity alignment "
                f"would be wrong"
            )
        sequences[chain.id] = "".join(letters)
    missing = set(CHAIN_ORDER) - set(sequences)
    if missing:
        raise ValueError(f"{pdb_path}: missing chain(s) {sorted(missing)}")
    return sequences


def cluster_members(directory: str) -> dict[str, list[str]]:
    """Map cluster id -> its model filenames, sorted by rank."""
    clusters: dict[str, list[str]] = {}
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".pdb"):
            continue
        clusters.setdefault(filename.split("_rank")[0], []).append(filename)
    return clusters


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=os.path.join(root, "data"),
                        help="input data directory")
    parser.add_argument("--out", default=os.path.join(root, "affine_input"),
                        help="output bundle directory")
    parser.add_argument("--path-prefix", default="",
                        help="prepended to every template and alignment path "
                             "written into the tables; leave empty to run AFfine "
                             "from inside the bundle directory")
    parser.add_argument("--no-copy-templates", action="store_true",
                        help="reference the PDBs where they are instead of "
                             "copying them into the bundle")
    args = parser.parse_args()

    docking_root = os.path.join(args.data, "docking")
    if not os.path.isdir(docking_root):
        print(f"docking directory not found: {docking_root}", file=sys.stderr)
        return 1

    align_dir = os.path.join(args.out, "alignments")
    os.makedirs(align_dir, exist_ok=True)

    def relative(*parts: str) -> str:
        return args.path_prefix + "/".join(parts) if args.path_prefix \
            else "/".join(parts)

    target_rows = []
    n_alignments = 0

    for pair in config.COMPLEXES:
        directory = os.path.join(docking_root, pair.name)
        clusters = cluster_members(directory)
        if len(clusters) != 10:
            print(f"warning: {pair.name} has {len(clusters)} clusters, expected 10",
                  file=sys.stderr)

        # Target sequence: identical for every model of this complex, so take it
        # from the first and verify the rest against it.
        reference = chain_sequences(os.path.join(directory, sorted(
            f for f in os.listdir(directory) if f.endswith(".pdb"))[0]))
        chainseq = "/".join(reference[c] for c in CHAIN_ORDER)
        flat = chainseq.replace("/", "")
        length = len(flat)

        if reference["H"] != pair.antibody.heavy or reference["L"] != pair.antibody.light:
            raise ValueError(
                f"{pair.name}: chains H/L in the docked models do not match the "
                f"sequences in abgrem/config.py"
            )

        alignstring = ";".join(f"{i}:{i}" for i in range(length))

        template_out = os.path.join(args.out, "templates", pair.name)
        if not args.no_copy_templates:
            os.makedirs(template_out, exist_ok=True)

        for cluster, members in sorted(clusters.items()):
            if len(members) != 4:
                print(f"warning: {pair.name} {cluster} has {len(members)} members, "
                      f"expected 4", file=sys.stderr)

            rows = []
            for filename in members:
                source = os.path.join(directory, filename)
                seqs = chain_sequences(source)
                if "/".join(seqs[c] for c in CHAIN_ORDER) != chainseq:
                    raise ValueError(
                        f"{source}: sequence differs from the other models of "
                        f"{pair.name}; the identity alignment would be wrong"
                    )
                if args.no_copy_templates:
                    template_path = os.path.abspath(source)
                else:
                    shutil.copyfile(source, os.path.join(template_out, filename))
                    template_path = relative("templates", pair.name, filename)
                rows.append((template_path, alignstring, length, length, length, 1))

            antigen = f"grem{pair.grem}"
            stem = f"aln_{pair.antibody.name}_{antigen}_{cluster}"
            align_path = os.path.join(align_dir, f"{stem}.tsv")
            with open(align_path, "w") as handle:
                handle.write("\t".join(ALIGN_COLUMNS) + "\n")
                for row in rows:
                    handle.write("\t".join(str(v) for v in row) + "\n")
            n_alignments += 1

            target_rows.append(
                (
                    f"{antigen}_{pair.antibody.name}_{cluster}",
                    chainseq,
                    relative("alignments", f"{stem}.tsv"),
                )
            )

        print(f"{pair.label}: {len(clusters)} clusters, target_len={length} "
              f"(H {len(reference['H'])} / L {len(reference['L'])} / "
              f"G {len(reference['G'])})")

    targets_path = os.path.join(args.out, "targets.tsv")
    with open(targets_path, "w") as handle:
        handle.write("\t".join(TARGET_COLUMNS) + "\n")
        for row in target_rows:
            handle.write("\t".join(row) + "\n")

    print(f"\n{n_alignments} alignment files -> {align_dir}")
    print(f"{len(target_rows)} targets -> {targets_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
