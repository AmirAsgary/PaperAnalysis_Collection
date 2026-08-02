"""
Amino-acid statistics derived from the antibody repertoire alignments.

The released alignments are *query-anchored*: the first record is the antibody
of interest, it contains no gaps, and the alignment length therefore equals the
antibody chain length.  Column *k* of the alignment is position *k* of the
antibody sequence, with no index mapping required.  :func:`load_alignment`
enforces this invariant rather than assuming it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from Bio import SeqIO

from .config import STANDARD_AA

#: Maximum Shannon entropy over 20 amino acids, in bits (≈ 4.32).
MAX_ENTROPY = float(np.log2(len(STANDARD_AA)))


def load_alignment(path: str, expected_query: str | None = None) -> list[str]:
    """
    Read a query-anchored alignment and verify its invariants.

    Raises if the alignment is ragged, if the query record carries gaps, or if
    the query does not match ``expected_query``.  Any of these would silently
    shift every position-specific score by an unknown offset.
    """
    sequences = [str(record.seq) for record in SeqIO.parse(path, "fasta")]
    if not sequences:
        raise ValueError(f"alignment is empty: {path}")

    length = len(sequences[0])
    ragged = [i for i, s in enumerate(sequences) if len(s) != length]
    if ragged:
        raise ValueError(
            f"{path}: {len(ragged)} records differ in length from the query "
            f"({length} columns); the alignment is not rectangular"
        )

    query = sequences[0]
    if "-" in query:
        raise ValueError(
            f"{path}: the query record contains gaps, so alignment columns do "
            f"not correspond to antibody sequence positions"
        )
    if expected_query is not None and query != expected_query:
        raise ValueError(
            f"{path}: query record does not match the configured antibody "
            f"sequence ({len(query)} vs {len(expected_query)} residues)"
        )
    return sequences


def _frequency_matrix(sequences: list[str]) -> np.ndarray:
    """Per-column amino-acid frequencies, gaps excluded from the denominator."""
    length = len(sequences[0])
    matrix = np.zeros((len(STANDARD_AA), length))
    for column in range(length):
        residues = [s[column] for s in sequences if s[column] in STANDARD_AA]
        if residues:
            counts = np.array([residues.count(aa) for aa in STANDARD_AA], dtype=float)
            matrix[:, column] = counts / len(residues)
    return matrix


def amino_acid_frequencies(sequences: list[str]) -> pd.DataFrame:
    """Observed frequency f_k(j): 20 amino acids × alignment columns."""
    matrix = _frequency_matrix(sequences)
    return pd.DataFrame(
        matrix,
        index=list(STANDARD_AA),
        columns=[f"Pos_{i + 1}" for i in range(matrix.shape[1])],
    )


def conservation(sequences: list[str]) -> np.ndarray:
    """
    Per-column conservation, ``1 - H(i) / H_max``.

    ``H(i)`` is the Shannon entropy in bits over the 20 standard amino acids.
    A fully conserved column scores 1, a uniformly variable column scores 0.
    """
    matrix = _frequency_matrix(sequences)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log2(matrix, where=matrix > 0)
    entropy = -np.nansum(np.where(matrix > 0, matrix * logs, 0.0), axis=0)
    return 1.0 - entropy / MAX_ENTROPY
