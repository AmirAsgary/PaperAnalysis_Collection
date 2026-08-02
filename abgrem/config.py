"""
Study configuration: antibodies, antigens, epitope, and all scoring constants.

Every tunable number used anywhere in the pipeline is declared here, so that the
manuscript methods section and the code can be checked against each other in one
place.
"""
from __future__ import annotations

from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# Epitope (experimentally defined by ELISA; residues 29–45 of the mature GREM)
# ─────────────────────────────────────────────────────────────────────────────
EPITOPE_RESIDUE_IDS = list(range(29, 46))
EPITOPE_SEQUENCE = {
    1: "EEGCNSRTIINRFCYGQ",
    2: "EEGCRSRTILNRFCYGQ",
}
ANTIGEN_CHAIN = "G"

#: Distance from an epitope Cα to any atom of an antibody residue, below which
#: the antibody residue is counted as an interaction partner (Å).
CONTACT_RADIUS = 5.5


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — position selection:  R_i = P_i · D_i · C_i · F_i
# ─────────────────────────────────────────────────────────────────────────────
#: Positions with P_i above this are carried into the ranking.
P_THRESHOLD = 0.5

#: Distance thresholds of the step function used to build D_i (Å).
DISTANCE_THRESHOLDS = (4.0, 5.0, 6.0, 7.0, 8.0, 12.0)

#: Steepness of the conservation sigmoid, and its offset.
CONSERVATION_ALPHA = 2.5
CONSERVATION_OFFSET = 1.0
CONSERVATION_EPSILON = 1e-9

#: Decay constant of the wild-type frequency penalty F_i = exp(-k · f_wt).
WT_FREQUENCY_DECAY = 15.0


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — mutation ranking (PSBDM):  S_k(i,j) = B(i,j) + w · P(k,j)
# ─────────────────────────────────────────────────────────────────────────────
PSBDM_WEIGHT = 1.0
PSBDM_PSEUDOCOUNT = 0.01

#: Uniform background frequency q_j, deliberately flat (1/20) so that the
#: MSA term stays independent of the BLOSUM62 background model.
BACKGROUND_FREQUENCY = 0.05

STANDARD_AA = tuple("ACDEFGHIKLMNPQRSTVWY")


# ─────────────────────────────────────────────────────────────────────────────
# Antibodies
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Antibody:
    """One antibody, independent of which antigen it is docked against."""

    name: str
    label: str  # display name used in figure titles
    heavy: str
    light: str

    @property
    def chains(self) -> dict[str, str]:
        return {"heavy": self.heavy, "light": self.light}


@dataclass(frozen=True)
class Complex:
    """One antibody–antigen pair."""

    antibody: Antibody
    grem: int

    @property
    def name(self) -> str:
        return f"{self.antibody.name}_GREM{self.grem}"

    @property
    def label(self) -> str:
        return f"{self.antibody.label} · GREM{self.grem}"

    @property
    def epitope(self) -> str:
        return EPITOPE_SEQUENCE[self.grem]


HU_AGREM = Antibody(
    name="Hu-aGREM",
    label="Hu-αGREM",
    heavy=("EVQLVQSGPEVVKPGASVKVSCKASGYSFTGYYMHWVRQAPGQGLEWMGYFFPYSGFSNYAQKFQG"
           "RVTLTVDKSKSTAYMELSRLRSEDTATYYCARGGLGRGYFDVWGQGTLVTVSS"),
    light=("DIQMTQSPSSLSASLGDRVTITCKASDHINNWLAWYQQKPGKAPRLLISGATSLETGVPSRFSGSG"
           "SGTDYTLTISSLQPEDVATYYCQQYWSSPRTFGGGTKLEIK"),
)

MU_AGREM = Antibody(
    name="Mu-aGREM",
    label="Mu-αGREM",
    heavy=("EVQLQQSGPELVKPGASVKISCKASGYSFTGYYMHWVKQSHGNILDWIGYFFPYNGFSNCNQKFKG"
           "KATLTVDKSSSTAYMELRSLTSEDSAVYYCARGGLGRGYFDVWGTGTTVTVSS"),
    light=("DIQMTQSPSYLSVSLGGRVTITCKASDHINNWLAWYQQKPGNAPRLLISGATSLETGVPSRFSGSG"
           "SGKDYTLSITSLQTEDVATYYCQQYWSSPRTFGGGTKLEIK"),
)

ANTIBODIES = (HU_AGREM, MU_AGREM)
COMPLEXES = tuple(Complex(ab, grem) for ab in ANTIBODIES for grem in (1, 2))

#: Positions carried into the focused mutation heat map of the manuscript.
#: (chain, 1-based position in that chain).
HIGHLIGHT_POSITIONS = {
    "Hu-aGREM": (("heavy", 31), ("heavy", 52), ("heavy", 57),
                 ("heavy", 101), ("light", 92)),
}
