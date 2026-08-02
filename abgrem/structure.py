"""
Structural analysis of the docked antibody–antigen complexes.

All released models carry three chains — H (heavy), L (light) and G (antigen) —
each numbered from 1, so chain H residue *i* is position *i* of the heavy-chain
sequence.
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import scipy.spatial
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import is_aa

from .config import (
    ANTIGEN_CHAIN,
    CONTACT_RADIUS,
    DISTANCE_THRESHOLDS,
    EPITOPE_RESIDUE_IDS,
)

_PARSER = PDBParser(QUIET=True)

#: Atom preference order when reducing a residue to a single representative point.
_REPRESENTATIVE_ATOMS = ("CA", "CB", "N", "O", "C")


def cluster_models(directory: str) -> list[str]:
    """Every docked model in a complex directory, sorted by cluster then rank."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".pdb")
    )


def cluster_representatives(directory: str) -> list[str]:
    """
    The rank-1 model of each HADDOCK cluster.

    These are the structures that define D_i: one representative per binding
    orientation, so that a heavily populated cluster does not dominate the
    distance score simply by contributing more models.
    """
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith("_rank1.pdb")
    )


def _residue_point(residue) -> np.ndarray:
    for atom_name in _REPRESENTATIVE_ATOMS:
        if atom_name in residue:
            return residue[atom_name].coord
    raise ValueError(f"no representative atom found for residue {residue}")


# ─────────────────────────────────────────────────────────────────────────────
# D_i — proximity of each antibody position to the epitope
# ─────────────────────────────────────────────────────────────────────────────
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def distance_score(pdb_path: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Graded proximity of every antibody residue to the epitope, for one model.

    For antibody residue *i* and epitope residue *k*, a step function counts how
    many of the thresholds in :data:`DISTANCE_THRESHOLDS` the pair falls within,
    normalised to [0, 1].  Summing over the epitope and passing the result
    through ``sigmoid(x) - 0.5`` maps "no contact at all" to exactly 0 and
    saturates at 0.5 for residues buried in the interface.

    Returns the per-chain score vectors and the underlying, un-summed
    antibody-residue × epitope-residue matrices.
    """
    structure = _PARSER.get_structure("model", pdb_path)

    coordinates: dict[str, list[np.ndarray]] = {}
    epitope_points: list[np.ndarray] = []
    for chain in structure[0]:
        points = []
        for residue in chain:
            point = _residue_point(residue)
            if chain.id == ANTIGEN_CHAIN and residue.id[1] in EPITOPE_RESIDUE_IDS:
                epitope_points.append(point)
            points.append(point)
        coordinates[chain.id] = points

    if len(epitope_points) != len(EPITOPE_RESIDUE_IDS):
        raise ValueError(
            f"{pdb_path}: found {len(epitope_points)} of "
            f"{len(EPITOPE_RESIDUE_IDS)} epitope residues in chain {ANTIGEN_CHAIN}"
        )

    scores: dict[str, np.ndarray] = {}
    matrices: dict[str, np.ndarray] = {}
    epitope = np.asarray(epitope_points)
    for chain_id, points in coordinates.items():
        if chain_id == ANTIGEN_CHAIN:
            continue
        distances = scipy.spatial.distance.cdist(np.asarray(points), epitope)
        steps = np.stack(
            [(distances <= threshold).astype(float) for threshold in DISTANCE_THRESHOLDS],
            axis=-1,
        )
        graded = steps.sum(axis=-1) / len(DISTANCE_THRESHOLDS)
        matrices[chain_id] = graded
        scores[chain_id] = np.clip(_sigmoid(graded.sum(axis=-1)) - 0.5, 0.0, 1.0)
    return scores, matrices


def mean_distance_score(pdb_paths: list[str]) -> dict[str, np.ndarray]:
    """D_i averaged over a set of models (one per HADDOCK cluster)."""
    if not pdb_paths:
        raise ValueError("no structures given for the distance score")
    accumulated: dict[str, list[np.ndarray]] = {}
    for path in pdb_paths:
        scores, _ = distance_score(path)
        for chain_id, vector in scores.items():
            accumulated.setdefault(chain_id, []).append(vector)
    return {chain: np.mean(np.stack(v, axis=-1), axis=-1)
            for chain, v in accumulated.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Epitope-side contacts
# ─────────────────────────────────────────────────────────────────────────────
def contacting_residues(pdb_path: str, epitope_residue_id: int,
                        radius: float = CONTACT_RADIUS) -> list[str]:
    """
    Antibody residues in contact with one epitope residue.

    A residue counts as a partner when any of its atoms lies within ``radius``
    of the **Cα atom** of the epitope residue.  The criterion is anchored on the
    epitope Cα rather than on a heavy-atom minimum distance, so it reports which
    antibody residues face a given epitope position rather than which atoms
    touch.
    """
    structure = _PARSER.get_structure("model", pdb_path)
    model = structure[0]
    target = model[ANTIGEN_CHAIN][(" ", epitope_residue_id, " ")]
    if "CA" not in target:
        raise ValueError(
            f"{pdb_path}: epitope residue {epitope_residue_id} has no Cα atom"
        )

    atoms = [a for a in structure.get_atoms() if is_aa(a.get_parent(), standard=True)]
    partners = set()
    for atom in NeighborSearch(atoms).search(target["CA"].coord, radius):
        residue = atom.get_parent()
        chain = residue.get_parent()
        if chain.id != ANTIGEN_CHAIN and is_aa(residue):
            partners.add(f"{chain.id}_{residue.id[1]}_{residue.resname}")
    return sorted(partners)


def epitope_contact_counts(pdb_paths: list[str]) -> dict[int, Counter]:
    """
    How often each antibody residue contacts each epitope position.

    Counts are pooled over every docked model supplied, so a value of *n* means
    the antibody residue was within :data:`CONTACT_RADIUS` of that epitope Cα in
    *n* of the models.
    """
    counts = {res_id: Counter() for res_id in EPITOPE_RESIDUE_IDS}
    for path in pdb_paths:
        for res_id in EPITOPE_RESIDUE_IDS:
            counts[res_id].update(contacting_residues(path, res_id))
    return counts
