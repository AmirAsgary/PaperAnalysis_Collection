"""
Every figure in the manuscript, plus the CSV behind each one.

Each function writes ``<name>.png`` (and ``.pdf`` where the figure is a main
panel) together with ``<name>.csv`` holding the exact plotted values, so no
figure depends on rerunning the pipeline to be re-drawn or checked.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import P_THRESHOLD, STANDARD_AA

GREM1_COLOUR = "#2f6f9f"
GREM2_COLOUR = "#e07a5f"


def _prepare(path: str) -> str:
    """Ensure the directory holding ``path`` exists, and return ``path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _save(fig, path: str, pdf: bool = False) -> None:
    _prepare(path)
    fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
    if pdf:
        fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def epitope_logo(matrix: pd.DataFrame, title: str, path: str) -> None:
    """
    Sequence logo of the epitope, letter height = mean number of antibody
    residues contacting that position per docked model.
    """
    import logomaker

    matrix.to_csv(f"{_prepare(path)}.csv")
    fig, ax = plt.subplots(figsize=(8, 3))
    logo = logomaker.Logo(matrix, color_scheme="NajafabadiEtAl2017", ax=ax)
    logo.ax.set_ylabel("Mean contacts per model")
    logo.ax.set_xlabel("Antigen residue number")
    logo.ax.set_title(title, fontsize=12)
    logo.style_spines(visible=False)
    logo.style_spines(spines=["left", "bottom"], visible=True)
    logo.style_xticks(anchor=0, spacing=1, rotation=90)
    _save(fig, path, pdf=True)


def epitope_contact_profile(counts: pd.DataFrame, title: str, path: str) -> None:
    """Mean contacts per model along the epitope, as a line profile."""
    counts.to_csv(f"{_prepare(path)}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(counts["epitope_position"], counts["mean_contacts_per_model"],
            marker="o", color="black", linewidth=1.2)
    ax.set_xlabel("Epitope residue")
    ax.set_ylabel("Mean contacts per model")
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", rotation=90)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
def stage1_profile(table: pd.DataFrame, title: str, path: str, annotate: int = 10) -> None:
    """P_i and R_i along the chain, with the top positions labelled."""
    table.to_csv(f"{_prepare(path)}.csv")
    selected = table["selected"]
    x = np.arange(len(table))

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(table) * 0.13), 6), sharex=True)
    axes[0].bar(x, table["P_i"], width=1.0,
                color=[GREM2_COLOUR if s else "#d9d9d9" for s in selected])
    axes[0].axhline(P_THRESHOLD, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("$P_i$")
    axes[0].set_title(f"{title} — Stage 1", fontsize=12)

    axes[1].bar(x, table["R_i"], width=1.0,
                color=[GREM1_COLOUR if s else "#d9d9d9" for s in selected])
    axes[1].set_ylabel("$R_i = P_i \\cdot D_i \\cdot C_i \\cdot F_i$")
    axes[1].set_xlabel("Residue position")

    for label in table[selected].nlargest(annotate, "R_i").index:
        position = int(label.split("_")[1])
        axes[1].annotate(
            f"{table.loc[label, 'residue']}{position}",
            xy=(position - 1, table.loc[label, "R_i"]),
            xytext=(0, 6), textcoords="offset points",
            fontsize=6, ha="center", fontweight="bold",
        )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def psbdm_heatmap(psbdm_table: pd.DataFrame, stage1_table: pd.DataFrame,
                  title: str, path: str, top_n: int = 10) -> None:
    """PSBDM scores for the highest-ranked positions."""
    top = stage1_table[stage1_table["selected"]].nlargest(top_n, "R_i").index
    columns = [p for p in top if p in psbdm_table.columns]
    block = psbdm_table[columns].rename(
        columns={p: f"{stage1_table.loc[p, 'residue']}{p.split('_')[1]}" for p in columns}
    )
    block.to_csv(f"{_prepare(path)}.csv")

    fig, ax = plt.subplots(figsize=(max(7, len(block.columns) * 1.1), 7))
    sns.heatmap(block, cmap="RdYlGn_r", annot=True, fmt=".1f", linewidths=0.5,
                ax=ax, cbar_kws={"label": "PSBDM score (lower = more disruptive)"})
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Candidate residue")
    ax.set_xlabel("Position (wild type)")
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
def grem1_vs_grem2(grem1: pd.DataFrame, grem2: pd.DataFrame,
                   title: str, path: str) -> None:
    """
    R_i for the same antibody against GREM1 and GREM2.

    P_i, C_i and F_i depend only on the antibody, so the two series differ
    exclusively through D_i, i.e. through docking geometry.
    """
    n = min(len(grem1), len(grem2))
    data = pd.DataFrame(
        {
            "position": [f"Pos_{i + 1}" for i in range(n)],
            "residue": grem1["residue"].values[:n],
            "GREM1_R_i": grem1["R_i"].values[:n],
            "GREM2_R_i": grem2["R_i"].values[:n],
        }
    )
    data.to_csv(f"{_prepare(path)}.csv", index=False)

    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.14), 4.5))
    ax.bar(x - 0.2, data["GREM1_R_i"], 0.4, label="GREM1",
           color=GREM1_COLOUR, edgecolor="black", linewidth=0.3)
    ax.bar(x + 0.2, data["GREM2_R_i"], 0.4, label="GREM2",
           color=GREM2_COLOUR, edgecolor="black", linewidth=0.3)
    ticks = list(range(0, n, 5))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{data['residue'][i]}{i + 1}" for i in ticks],
                       rotation=90, fontsize=7)
    ax.set_xlabel("Position")
    ax.set_ylabel("$R_i$")
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path, pdf=True)


def top_positions_bar(grem1: pd.DataFrame, grem2: pd.DataFrame,
                      title: str, path: str, top_n: int = 10) -> None:
    """
    The highest-scoring paratope positions, GREM1 open and GREM2 filled.

    Positions are labelled ``<number>_<wild type>`` and the score is the
    paratoping score R_i.
    """
    positions = sorted(
        set(grem1[grem1["selected"]].index) | set(grem2[grem2["selected"]].index),
        key=lambda p: int(p.split("_")[1]),
    )
    if not positions:
        return
    rows = [
        {
            "position": f"{p.split('_')[1]}_{grem1.loc[p, 'residue']}",
            "GREM1_R_i": float(grem1.loc[p, "R_i"]),
            "GREM2_R_i": float(grem2.loc[p, "R_i"]),
        }
        for p in positions
    ]
    data = (pd.DataFrame(rows)
            .assign(_key=lambda d: d[["GREM1_R_i", "GREM2_R_i"]].max(axis=1))
            .sort_values("_key", ascending=False)
            .drop(columns="_key")
            .head(top_n)
            .reset_index(drop=True))
    data.to_csv(f"{_prepare(path)}.csv", index=False)

    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(max(5, len(data) * 0.5), 4))
    ax.bar(x - 0.2, data["GREM1_R_i"], 0.4, label="GREM1",
           color="white", edgecolor="black", linewidth=0.6)
    ax.bar(x + 0.2, data["GREM2_R_i"], 0.4, label="GREM2",
           color="black", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(data["position"], rotation=90, fontsize=9)
    ax.set_ylabel("Paratoping Score ($R_i$)")
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path, pdf=True)


def mutation_heatmap(psbdm_by_chain: dict[str, pd.DataFrame],
                     stage1_by_chain: dict[str, pd.DataFrame],
                     positions: tuple[tuple[str, int], ...],
                     title: str, path: str) -> None:
    """
    PSBDM scores at the positions carried forward as mutation candidates.

    The wild-type residue of each column is masked and crossed out.
    """
    columns, wild_types, chains = [], [], []
    for chain, number in positions:
        key = f"Pos_{number}"
        psbdm_table = psbdm_by_chain[chain]
        if key not in psbdm_table.columns:
            continue
        wild_type = stage1_by_chain[chain].loc[key, "residue"]
        columns.append(psbdm_table[key].reindex(list(STANDARD_AA)).astype(float).values)
        wild_types.append(wild_type)
        chains.append(chain)
    if not columns:
        return

    labels = [f"{wt}{number}" for wt, (_, number) in zip(wild_types, positions)]
    data = pd.DataFrame(np.column_stack(columns), index=list(STANDARD_AA), columns=labels)
    data.to_csv(f"{_prepare(path)}.csv")

    mask = np.zeros(data.shape, dtype=bool)
    for j, wild_type in enumerate(wild_types):
        mask[list(STANDARD_AA).index(wild_type), j] = True
    n_heavy = sum(1 for c in chains if c == "heavy")

    for cmap, suffix in (("RdYlGn_r", "green_red"), ("RdBu_r", "blue_red")):
        fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.1), 7))
        sns.heatmap(data, cmap=cmap, linewidths=0.8, linecolor="white", ax=ax,
                    mask=mask,
                    cbar_kws={"label": "PSBDM score (lower = more disruptive)",
                              "shrink": 0.7})
        for j, wild_type in enumerate(wild_types):
            i = list(STANDARD_AA).index(wild_type)
            ax.plot([j, j + 1], [i, i + 1], color="black", linewidth=1.5)
            ax.plot([j, j + 1], [i + 1, i], color="black", linewidth=1.5)
        if 0 < n_heavy < len(labels):
            ax.axvline(x=n_heavy, color="black", linewidth=2)
        if n_heavy:
            ax.text(n_heavy / 2, -0.8, "heavy chain", ha="center",
                    va="bottom", fontsize=10, fontweight="bold")
        if n_heavy < len(labels):
            ax.text(n_heavy + (len(labels) - n_heavy) / 2, -0.8, "light chain",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, pad=24)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", rotation=0)
        _save(fig, f"{path}_{suffix}", pdf=True)


def model_confidence(table: pd.DataFrame, path: str) -> None:
    """
    Supplementary: confidence of the per-cluster AlphaFold models.

    One point per cluster, in four panels: pLDDT and PAE over the whole model,
    and the same two quantities restricted to the antibody–antigen interface.
    """
    table.to_csv(f"{_prepare(path)}.csv", index=False)

    panels = (
        ("plddt", "pLDDT", "Whole model"),
        ("interface_plddt", "Interface pLDDT", "Interface"),
        ("pae", "PAE (Å)", "Whole model"),
        ("interface_pae", "Interface PAE (Å)", "Interface"),
    )
    order = sorted(table["complex"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (column, label, scope) in zip(axes.ravel(), panels):
        sns.boxplot(data=table, x="complex", y=column, order=order,
                    color="white", width=0.55, fliersize=0, ax=ax)
        for line in ax.lines:
            line.set_color("black")
        for patch in ax.patches:
            patch.set_edgecolor("black")
            patch.set_facecolor("white")
        sns.stripplot(data=table, x="complex", y=column, order=order,
                      color="black", size=4, alpha=0.6, jitter=0.18, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_title(scope, fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("AlphaFold confidence of the per-cluster models", fontsize=13)
    fig.tight_layout()
    _save(fig, path, pdf=True)
