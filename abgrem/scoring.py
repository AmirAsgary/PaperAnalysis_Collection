"""
The two-stage scoring scheme.

Stage 1 selects positions on the antibody that are worth mutating;
Stage 2 ranks the 19 possible substitutions at each of those positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BACKGROUND_FREQUENCY,
    CONSERVATION_ALPHA,
    CONSERVATION_EPSILON,
    CONSERVATION_OFFSET,
    P_THRESHOLD,
    PSBDM_PSEUDOCOUNT,
    PSBDM_WEIGHT,
    STANDARD_AA,
    WT_FREQUENCY_DECAY,
)

# ─────────────────────────────────────────────────────────────────────────────
# BLOSUM62
# ─────────────────────────────────────────────────────────────────────────────
_BLOSUM_ORDER = tuple("ARNDCQEGHILKMFPSTWYVBZX")
_BLOSUM_ROWS = {
    "A": (4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0),
    "R": (-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1),
    "N": (-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1),
    "D": (-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1),
    "C": (0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2),
    "Q": (-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1),
    "E": (-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1),
    "G": (0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1),
    "H": (-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1),
    "I": (-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1),
    "L": (-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1),
    "K": (-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1),
    "M": (-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1),
    "F": (-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1),
    "P": (-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2),
    "S": (1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0),
    "T": (0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0),
    "W": (-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2),
    "Y": (-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1),
    "V": (0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1),
    "B": (-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1),
    "Z": (-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1),
    "X": (0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1),
}


def blosum62(wild_type: str, mutant: str) -> int:
    """Standard BLOSUM62 substitution score B(i, j)."""
    return _BLOSUM_ROWS[wild_type][_BLOSUM_ORDER.index(mutant)]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 terms
# ─────────────────────────────────────────────────────────────────────────────
def interaction_probability(proabc2: pd.DataFrame) -> np.ndarray:
    """
    P_i — probability that position *i* makes any contact with the antigen.

    ProABC-2 reports three independent probabilities per position: general (pt),
    hydrogen-bond (hb) and hydrophobic (hy).  They are combined as the
    complement of making none of the three::

        P_i = 1 - (1 - pt)(1 - hb)(1 - hy)
    """
    absent = 1.0 - proabc2.loc[["pt", "hy", "hb"], :].to_numpy(dtype=float)
    return 1.0 - np.prod(absent, axis=0)


def conservation_score(conservation: np.ndarray) -> np.ndarray:
    """
    C_i — preference for hypervariable positions.

    A sigmoid of ``-log(conservation)`` that approaches 1 at variable positions
    and 0 at framework positions, so that conserved residues are effectively
    removed from consideration.
    """
    inner = CONSERVATION_ALPHA * (
        -np.log(np.asarray(conservation, dtype=float) + CONSERVATION_EPSILON)
        - CONSERVATION_OFFSET
    )
    return 1.0 / (1.0 + np.exp(-inner))


def wildtype_frequency_penalty(sequence: str, frequencies: pd.DataFrame) -> np.ndarray:
    """
    F_i — penalty for positions whose wild-type residue is the repertoire consensus.

    ``F_i = exp(-15 · f_wt)`` falls to 0.22 at a wild-type frequency of 10 % and
    to 0.01 at 30 %, so consensus positions are effectively excluded.
    """
    return np.array(
        [
            np.exp(-WT_FREQUENCY_DECAY * float(frequencies.loc[aa, f"Pos_{i + 1}"]))
            for i, aa in enumerate(sequence)
        ],
        dtype=float,
    )


def stage1(sequence: str, p_i: np.ndarray, d_i: np.ndarray,
           c_i: np.ndarray, f_i: np.ndarray,
           p_threshold: float = P_THRESHOLD) -> pd.DataFrame:
    """
    Combine the four terms into ``R_i = P_i · D_i · C_i · F_i`` and select
    positions above the interaction-probability threshold.
    """
    lengths = {len(sequence), len(p_i), len(d_i), len(c_i), len(f_i)}
    if len(lengths) != 1:
        raise ValueError(
            f"stage 1 inputs disagree in length: sequence={len(sequence)}, "
            f"P_i={len(p_i)}, D_i={len(d_i)}, C_i={len(c_i)}, F_i={len(f_i)}"
        )

    table = pd.DataFrame(
        {
            "residue": list(sequence),
            "P_i": p_i,
            "D_i": d_i,
            "C_i": c_i,
            "F_i": f_i,
            "R_i": p_i * d_i * c_i * f_i,
        },
        index=[f"Pos_{i}" for i in range(1, len(sequence) + 1)],
    )
    table["selected"] = table["P_i"] > p_threshold
    table["rank"] = table.loc[table["selected"], "R_i"].rank(ascending=False, method="min")
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Position-Specific BLOSUM Dissimilarity Matrix
# ─────────────────────────────────────────────────────────────────────────────
def position_specific_msa_score(frequencies: pd.DataFrame,
                                pseudocount: float = PSBDM_PSEUDOCOUNT) -> pd.DataFrame:
    """
    P(k, j) — log-odds enrichment of amino acid *j* at position *k*::

        P(k, j) = 2 · log2( (f_k(j) + ε) / q_j )

    The background q_j is deliberately uniform (1/20), which keeps this term
    independent of the background model already built into BLOSUM62.  Values are
    positive for residues over-represented at the position and negative for
    residues the repertoire avoids there.
    """
    observed = frequencies.reindex(list(STANDARD_AA)).fillna(0.0).to_numpy(dtype=float)
    scores = 2.0 * np.log2((observed + pseudocount) / BACKGROUND_FREQUENCY)
    return pd.DataFrame(scores, index=list(STANDARD_AA), columns=frequencies.columns)


def psbdm(sequence: str, frequencies: pd.DataFrame,
          weight: float = PSBDM_WEIGHT,
          pseudocount: float = PSBDM_PSEUDOCOUNT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Position-Specific BLOSUM Dissimilarity Matrix::

        S_k(i, j) = B(i, j) + w · P(k, j)

    Lower scores mark substitutions that are both chemically dissimilar to the
    wild type and rare at that position in the repertoire — the mutations
    expected to perturb binding most.  The wild-type residue itself is NaN.
    """
    msa_term = position_specific_msa_score(frequencies, pseudocount=pseudocount)

    columns = {}
    for index, wild_type in enumerate(sequence):
        column = f"Pos_{index + 1}"
        if column not in msa_term.columns:
            raise ValueError(f"position {column} is missing from the alignment")
        columns[column] = [
            np.nan if mutant == wild_type
            else blosum62(wild_type, mutant) + weight * msa_term.loc[mutant, column]
            for mutant in STANDARD_AA
        ]
    return pd.DataFrame(columns, index=list(STANDARD_AA)), msa_term


def rank_mutations(stage1_table: pd.DataFrame, psbdm_table: pd.DataFrame,
                   chain: str, top_n: int = 5) -> pd.DataFrame:
    """
    For every selected position, list the ``top_n`` most disruptive substitutions.

    Positions are ordered by R_i.  Cysteine is reported when it ranks highly but
    is not removed here; the manuscript candidates were chosen by stepping to the
    next-best substitution wherever cysteine came first, to avoid unpaired
    thiols during expression.
    """
    rows = []
    selected = stage1_table[stage1_table["selected"]].sort_values("R_i", ascending=False)
    for position in selected.index:
        if position not in psbdm_table.columns:
            continue
        scores = psbdm_table[position].dropna()
        if scores.empty:
            continue
        best = scores.nsmallest(top_n)
        rows.append(
            {
                "position": position,
                "chain": chain,
                "wild_type": stage1_table.loc[position, "residue"],
                "P_i": round(float(stage1_table.loc[position, "P_i"]), 4),
                "D_i": round(float(stage1_table.loc[position, "D_i"]), 4),
                "C_i": round(float(stage1_table.loc[position, "C_i"]), 4),
                "F_i": round(float(stage1_table.loc[position, "F_i"]), 4),
                "R_i": round(float(stage1_table.loc[position, "R_i"]), 4),
                "best_mutation": best.index[0],
                "best_score": round(float(best.iloc[0]), 3),
                "top_mutations": ",".join(best.index),
                "top_scores": ",".join(f"{v:.3f}" for v in best.values),
            }
        )
    return pd.DataFrame(rows)
