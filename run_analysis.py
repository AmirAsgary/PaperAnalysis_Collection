#!/usr/bin/env python3
"""
Reproduce the complete analysis: epitope contacts, paratope scoring, mutation
ranking, and every manuscript figure.

    python run_analysis.py

Reads everything from ``data/`` and writes everything to ``results/``.  The run
is deterministic: repeating it reproduces byte-identical tables.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from abgrem import config, figures, msa, scoring, structure

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_proabc2(path: str, length: int) -> pd.DataFrame:
    """ProABC-2 per-residue probabilities, transposed to 3 rows × N positions."""
    table = pd.read_csv(path).transpose()
    if table.shape[1] != length:
        raise ValueError(
            f"{path}: {table.shape[1]} positions, expected {length} to match "
            f"the antibody sequence"
        )
    table.columns = [f"Pos_{i}" for i in range(1, length + 1)]
    return table


def load_antibody(data_dir: str, antibody: config.Antibody) -> dict:
    """Alignments and ProABC-2 predictions for one antibody, with checks."""
    base = os.path.join(data_dir, "antibodies", antibody.name)
    loaded = {}
    for chain, sequence in antibody.chains.items():
        alignment = msa.load_alignment(
            os.path.join(base, f"msa_{chain}.fa"), expected_query=sequence
        )
        loaded[chain] = {
            "sequence": sequence,
            "frequencies": msa.amino_acid_frequencies(alignment),
            "conservation": msa.conservation(alignment),
            "n_sequences": len(alignment),
            "proabc2": load_proabc2(
                os.path.join(base, f"proabc2_{chain}.csv"), len(sequence)
            ),
        }
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# Epitope-side analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_epitope(pair: config.Complex, models: list[str], outdir: str) -> pd.DataFrame:
    """
    Contacts made by each epitope position, pooled over every docked model.

    Writes one table per epitope residue listing its antibody partners, plus a
    per-position summary, and returns the matrix used for the sequence logo.
    """
    os.makedirs(outdir, exist_ok=True)
    counts = structure.epitope_contact_counts(models)
    n_models = len(models)

    summary_rows = []
    for index, res_id in enumerate(config.EPITOPE_RESIDUE_IDS):
        residue = pair.epitope[index]
        partners = counts[res_id]
        label = f"{residue}{res_id}"

        partner_table = pd.DataFrame(
            {
                "antibody_residue": list(partners.keys()),
                "n_models": list(partners.values()),
                "fraction_of_models": [v / n_models for v in partners.values()],
            }
        ).sort_values("n_models", ascending=False)
        partner_table.to_csv(os.path.join(outdir, f"{label}_partners.csv"), index=False)

        total = sum(partners.values())
        heavy = sum(v for k, v in partners.items() if k.startswith("H"))
        summary_rows.append(
            {
                "epitope_position": label,
                "residue": residue,
                "residue_id": res_id,
                "n_distinct_partners": len(partners),
                "total_contacts": total,
                "mean_contacts_per_model": total / n_models,
                "fraction_heavy_chain": heavy / total if total else 0.0,
                "fraction_light_chain": (total - heavy) / total if total else 0.0,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(outdir, "epitope_summary.csv"), index=False)

    # Logo matrix: rows are epitope positions, columns the residue letters
    # present in the epitope; the height of each letter is the mean number of
    # antibody residues contacting that position per model.
    letters = sorted(set(pair.epitope))
    logo = pd.DataFrame(0.0, index=config.EPITOPE_RESIDUE_IDS, columns=letters)
    logo.index.name = "residue_id"
    for index, (res_id, residue) in enumerate(
        zip(config.EPITOPE_RESIDUE_IDS, pair.epitope)
    ):
        logo.loc[res_id, residue] = summary.loc[index, "mean_contacts_per_model"]
    return logo


# ─────────────────────────────────────────────────────────────────────────────
# Per-complex scoring
# ─────────────────────────────────────────────────────────────────────────────
def analyse_complex(pair: config.Complex, antibody_data: dict, data_dir: str,
                    results_dir: str, weight: float, pseudocount: float) -> dict:
    print(f"\n{pair.label}")
    docking_dir = os.path.join(data_dir, "docking", pair.name)
    models = structure.cluster_models(docking_dir)
    representatives = structure.cluster_representatives(docking_dir)
    print(f"  {len(models)} docked models, {len(representatives)} cluster representatives")

    logo = analyse_epitope(
        pair, models, os.path.join(results_dir, "epitope_contacts", pair.name)
    )
    figures.epitope_logo(
        logo, f"{pair.label} epitope",
        os.path.join(results_dir, "figures", "epitope_logos", pair.name),
    )
    summary = pd.read_csv(
        os.path.join(results_dir, "epitope_contacts", pair.name, "epitope_summary.csv")
    )
    figures.epitope_contact_profile(
        summary[["epitope_position", "mean_contacts_per_model"]],
        f"{pair.label} epitope contacts",
        os.path.join(results_dir, "figures", "epitope_profiles", pair.name),
    )

    distance = structure.mean_distance_score(representatives)
    chain_key = {"heavy": "H", "light": "L"}

    outcome = {}
    for chain in ("heavy", "light"):
        loaded = antibody_data[chain]
        sequence = loaded["sequence"]

        stage1 = scoring.stage1(
            sequence=sequence,
            p_i=scoring.interaction_probability(loaded["proabc2"]),
            d_i=distance[chain_key[chain]],
            c_i=scoring.conservation_score(loaded["conservation"]),
            f_i=scoring.wildtype_frequency_penalty(sequence, loaded["frequencies"]),
        )
        psbdm_table, msa_term = scoring.psbdm(
            sequence, loaded["frequencies"], weight=weight, pseudocount=pseudocount
        )
        mutations = scoring.rank_mutations(stage1, psbdm_table, chain[0].upper())

        scores_dir = os.path.join(results_dir, "scores")
        mutations_dir = os.path.join(results_dir, "mutations")
        os.makedirs(scores_dir, exist_ok=True)
        os.makedirs(mutations_dir, exist_ok=True)
        stem = f"{pair.name}_{chain}"
        stage1.to_csv(os.path.join(scores_dir, f"{stem}_stage1.csv"))
        psbdm_table.to_csv(os.path.join(mutations_dir, f"{stem}_PSBDM.csv"))
        msa_term.to_csv(os.path.join(mutations_dir, f"{stem}_msa_logodds.csv"))
        mutations.to_csv(os.path.join(mutations_dir, f"{stem}_ranked.csv"), index=False)

        figures.stage1_profile(
            stage1, f"{pair.label} — {chain} chain",
            os.path.join(results_dir, "figures", "stage1_profiles", stem),
        )
        figures.psbdm_heatmap(
            psbdm_table, stage1, f"{pair.label} — {chain} chain",
            os.path.join(results_dir, "figures", "psbdm_heatmaps", stem),
        )

        print(f"  {chain:<5} {int(stage1['selected'].sum()):>3} positions selected "
              f"(P_i > {config.P_THRESHOLD})")
        for _, row in mutations.head(3).iterrows():
            print(f"        {row['position']:<8} {row['wild_type']} → "
                  f"{row['best_mutation']}   R_i={row['R_i']:.3f}  "
                  f"PSBDM={row['best_score']:.2f}")

        outcome[chain] = {"stage1": stage1, "psbdm": psbdm_table, "mutations": mutations}
    return outcome


# ─────────────────────────────────────────────────────────────────────────────
# Cross-complex figures
# ─────────────────────────────────────────────────────────────────────────────
def comparison_figures(results: dict, results_dir: str) -> None:
    print("\nComparison figures")
    for antibody in config.ANTIBODIES:
        grem1 = results.get(f"{antibody.name}_GREM1")
        grem2 = results.get(f"{antibody.name}_GREM2")
        if not (grem1 and grem2):
            continue
        for chain in ("heavy", "light"):
            stem = f"{antibody.name}_{chain}"
            figures.grem1_vs_grem2(
                grem1[chain]["stage1"], grem2[chain]["stage1"],
                f"{antibody.label} — {chain} chain",
                os.path.join(results_dir, "figures", "grem1_vs_grem2", stem),
            )
            figures.top_positions_bar(
                grem1[chain]["stage1"], grem2[chain]["stage1"],
                f"{antibody.label} — {chain} chain",
                os.path.join(results_dir, "figures", "top_positions", stem),
            )
        print(f"  {antibody.label}: GREM1 vs GREM2, heavy and light")

    for name, positions in config.HIGHLIGHT_POSITIONS.items():
        pair = results.get(f"{name}_GREM1")
        if pair is None:
            continue
        antibody = next(a for a in config.ANTIBODIES if a.name == name)
        figures.mutation_heatmap(
            {c: pair[c]["psbdm"] for c in ("heavy", "light")},
            {c: pair[c]["stage1"] for c in ("heavy", "light")},
            positions,
            f"{antibody.label} — candidate mutations",
            os.path.join(results_dir, "figures", "mutation_candidates", name),
        )
        print(f"  {antibody.label}: candidate mutation heat map")


def template_benchmark_figure(data_dir: str, results_dir: str) -> pd.DataFrame | None:
    """
    Supplementary: the AlphaFold single- versus multi-template comparison that
    motivated using one docked template per prediction.

    The manuscript reports only the single-template protocol; this figure is
    kept as supporting evidence for that choice and is not a main-text panel.
    """
    bench_dir = os.path.join(data_dir, "alphafold", "template_benchmark")
    if not os.path.isdir(bench_dir):
        return None

    rows = []
    for filename in sorted(os.listdir(bench_dir)):
        if not filename.endswith(".tsv"):
            continue
        table = pd.read_csv(os.path.join(bench_dir, filename), sep="\t")
        plddt_columns = [c for c in table.columns
                         if c.endswith("_plddt") and not c.split("_plddt")[0][-1].isdigit()]
        for _, row in table.iterrows():
            for column in plddt_columns:
                rows.append(
                    {
                        "complex": f"{row['antibody']}·{row['antigen'].upper()}",
                        "targetid": row["targetid"],
                        "alphafold_params": row["alphafold_params"],
                        "n_templates": int(row["n_templates"]),
                        "plddt": float(row[column]),
                    }
                )
    if not rows:
        return None

    table = pd.DataFrame(rows)
    figures.template_benchmark(
        table,
        os.path.join(results_dir, "figures", "supplementary",
                     "alphafold_template_choice"),
    )
    stats = (table.groupby("n_templates")["plddt"]
             .agg(["count", "mean", "std", "min", "max"]).round(2))
    print("\nSupplementary — AlphaFold template choice (pLDDT)")
    print(stats.to_string())
    return stats


# ─────────────────────────────────────────────────────────────────────────────
def write_summary(results: dict, benchmark: pd.DataFrame | None,
                  results_dir: str) -> None:
    lines = [
        "=" * 72,
        "Paratope scoring and mutation prioritisation — summary",
        "=" * 72,
        f"Stage 1   R_i = P_i · D_i · C_i · F_i,  selection at P_i > {config.P_THRESHOLD}",
        f"          D_i thresholds {list(config.DISTANCE_THRESHOLDS)} Å, "
        f"averaged over cluster representatives",
        f"          F_i = exp(-{config.WT_FREQUENCY_DECAY:g} · f_wt)",
        f"Stage 2   S_k(i,j) = B(i,j) + {config.PSBDM_WEIGHT:g} · P(k,j),  "
        f"P(k,j) = 2·log2((f + {config.PSBDM_PSEUDOCOUNT:g}) / {config.BACKGROUND_FREQUENCY:g})",
        "          Lower PSBDM = more disruptive substitution",
        "",
    ]

    payload = {}
    for name, chains in results.items():
        lines.append(f"--- {name} ---")
        payload[name] = {}
        for chain in ("heavy", "light"):
            mutations = chains[chain]["mutations"]
            lines.append(f"  {chain}: {len(mutations)} selected positions")
            for _, row in mutations.head(5).iterrows():
                lines.append(
                    f"    {row['position']:<8} {row['wild_type']} → "
                    f"{row['best_mutation']}   R_i={row['R_i']:.3f}   "
                    f"PSBDM={row['best_score']:.2f}"
                )
            payload[name][chain] = {
                "n_selected": int(len(mutations)),
                "top": mutations.head(5)[
                    ["position", "wild_type", "best_mutation", "R_i", "best_score"]
                ].to_dict("records"),
            }
        lines.append("")

    if benchmark is not None:
        lines += ["Supplementary — AlphaFold template choice (pLDDT)",
                  benchmark.to_string(), ""]
        payload["alphafold_template_choice"] = benchmark.to_dict("index")

    lines.append("=" * 72)
    text = "\n".join(lines)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "SUMMARY.txt"), "w") as handle:
        handle.write(text + "\n")
    with open(os.path.join(results_dir, "summary.json"), "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print("\n" + text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="input data directory")
    parser.add_argument("--results", default="results", help="output directory")
    parser.add_argument("--weight", type=float, default=config.PSBDM_WEIGHT,
                        help="weight w of the MSA term in the PSBDM score")
    parser.add_argument("--pseudocount", type=float, default=config.PSBDM_PSEUDOCOUNT,
                        help="pseudocount ε in the MSA log-odds term")
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(f"data directory not found: {args.data}\n"
              f"Download and unpack the dataset first (see README.md).",
              file=sys.stderr)
        return 1

    antibody_data = {}
    for antibody in config.ANTIBODIES:
        antibody_data[antibody.name] = load_antibody(args.data, antibody)
        n = antibody_data[antibody.name]["heavy"]["n_sequences"]
        print(f"{antibody.label}: repertoire alignment of {n:,} sequences")

    results = {}
    for pair in config.COMPLEXES:
        results[pair.name] = analyse_complex(
            pair, antibody_data[pair.antibody.name], args.data, args.results,
            args.weight, args.pseudocount,
        )

    comparison_figures(results, args.results)
    benchmark = template_benchmark_figure(args.data, args.results)
    write_summary(results, benchmark, args.results)
    print(f"\nDone. All output in {args.results}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
