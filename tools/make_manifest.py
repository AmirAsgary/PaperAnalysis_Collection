#!/usr/bin/env python3
"""
Write ``data/MANIFEST.md``: a description of every directory in the released
dataset together with an MD5 checksum for each file, so a download can be
verified without rerunning anything.

    python tools/make_manifest.py --data data
"""
from __future__ import annotations

import argparse
import hashlib
import os

DESCRIPTIONS = {
    "antibodies": (
        "Antibody variable-domain sequences, ProABC-2 per-residue interaction "
        "probabilities, and the query-anchored repertoire alignments used for "
        "the conservation and frequency terms."
    ),
    "antigens": (
        "GREM1 (PDB 5AEJ) and GREM2 (PDB 5HK5) crystal structures, plus the "
        "AlphaFold3 models after removal of low-confidence regions."
    ),
    "docking": (
        "HADDOCK 2.4 models, one directory per antibody-antigen pair: the ten "
        "best-scoring clusters with their four best members each. Chains are H "
        "(heavy), L (light) and G (antigen), each numbered from 1. Used for the "
        "epitope-contact analysis, and as the templates for the AlphaFold models."
    ),
    "alphafold": (
        "One AlphaFold model per HADDOCK cluster, predicted from all four "
        "members of that cluster as templates, with its pLDDT and PAE arrays; "
        "these define the distance score D_i. Plus the Rosetta-relaxed "
        "structures used for visualisation."
    ),
}


def md5(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    args = parser.parse_args()

    entries = []
    for root, _, filenames in os.walk(args.data):
        for filename in sorted(filenames):
            if filename == "MANIFEST.md":
                continue
            full = os.path.join(root, filename)
            entries.append((os.path.relpath(full, args.data), os.path.getsize(full), full))
    entries.sort()

    total = sum(size for _, size, _ in entries)
    lines = [
        "# Dataset manifest",
        "",
        f"{len(entries)} files, {human(total)} unpacked.",
        "",
        "Verify a download with:",
        "",
        "```bash",
        "md5sum -c <(sed -n 's/^| `\\(.*\\)` | .* | `\\(.*\\)` |$/\\2  \\1/p' MANIFEST.md)",
        "```",
        "",
        "## Contents",
        "",
    ]
    for name, description in DESCRIPTIONS.items():
        directory = os.path.join(args.data, name)
        if not os.path.isdir(directory):
            continue
        count = sum(1 for e in entries if e[0].startswith(name + os.sep))
        size = sum(e[1] for e in entries if e[0].startswith(name + os.sep))
        lines += [f"### `{name}/` — {count} files, {human(size)}", "", description, ""]

    lines += ["## Checksums", "", "| File | Size | MD5 |", "|---|---|---|"]
    for name, size, full in entries:
        lines.append(f"| `{name}` | {human(size)} | `{md5(full)}` |")

    out = os.path.join(args.data, "MANIFEST.md")
    with open(out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"{out}: {len(entries)} files, {human(total)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
