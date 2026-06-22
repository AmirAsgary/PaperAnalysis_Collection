#!/usr/bin/env python3
"""
Two-Stage Paratope Scoring & Mutation Suggestion Pipeline
=========================================================
Stage 1 — Position Selection:
    Filter positions where P_i > 0.5, then rank by
    R_i = P_i * D_i * C_i * exp(-15 * Freq_wt_i).
Stage 2 — Mutation Ranking using Position-Specific BLOSUM Dissimilarity Matrix (PSBDM):
    For selected positions, rank all 19 possible substitutions using:
    Score(j) = B(i, j) + [w * P(k, j)]
    Where:
    - B(i, j) = BLOSUM62 score between wild-type i and mutant j
    - w = weight parameter (default 1.0)
    - P(k, j) = 2 * log2( (f_k(j) + pseudocount) / q_j )
    Lower scores indicate BETTER mutations (more dissimilar + rare at that position).

Optional preprocessing (--clean flag):
    Split HADDOCK-docked PDBs into H/L/G chains, compute per-epitope
    structural proximity CSVs, and plot interaction frequencies.

Usage:
    python run_analysis.py              # scoring + logos + figures
    python run_analysis.py --clean      # also re-run PDB cleaning
"""
import os, sys, math, warnings, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import scripts
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# EPITOPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════
EPITOPE_IDS = list(range(29, 46))  # residue numbers 29–45
EPITOPE_GREM1 = list('EEGCNSRTIINRFCYGQ')
EPITOPE_GREM2 = list('EEGCRSRTILNRFCYGQ')
CHAIN_ID = 'G'  # GREM chain ID after renaming

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — all antibody–GREM pairs
# ═══════════════════════════════════════════════════════════════════
ANTIBODIES = [
    {'ab': 'huvar7_grem1', 'pdb_folder': 'alphafold/templates/huvar7_grem1/',
     'other_folder': 'Hu-Var-7',
     'haddock_folder': 'Hu-Var-7/395611-HuVar_GREM1_summary',
     'ab_chain': 'B', 'grem_chain': 'A', 'grem': 1,
     'sequenceh': 'EVQLVQSGPEVVKPGASVKVSCKASGYSFTGYYMHWVRQAPGQGLEWMGYFFPYSGFSNYAQKFQGRVTLTVDKSKSTAYMELSRLRSEDTATYYCARGGLGRGYFDVWGQGTLVTVSS',
     'sequencel': 'DIQMTQSPSSLSASLGDRVTITCKASDHINNWLAWYQQKPGKAPRLLISGATSLETGVPSRFSGSGSGTDYTLTISSLQPEDVATYYCQQYWSSPRTFGGGTKLEIK'},
    {'ab': 'huvar7_grem2', 'pdb_folder': 'alphafold/templates/huvar7_grem2/',
     'other_folder': 'Hu-Var-7',
     'haddock_folder': 'Hu-Var-7/395612-HuVar_GREM2_summary',
     'ab_chain': 'B', 'grem_chain': 'A', 'grem': 2,
     'sequenceh': 'EVQLVQSGPEVVKPGASVKVSCKASGYSFTGYYMHWVRQAPGQGLEWMGYFFPYSGFSNYAQKFQGRVTLTVDKSKSTAYMELSRLRSEDTATYYCARGGLGRGYFDVWGQGTLVTVSS',
     'sequencel': 'DIQMTQSPSSLSASLGDRVTITCKASDHINNWLAWYQQKPGKAPRLLISGATSLETGVPSRFSGSGSGTDYTLTISSLQPEDVATYYCQQYWSSPRTFGGGTKLEIK'},
    {'ab': '3a13_grem1', 'pdb_folder': 'alphafold/templates/3a13_grem1/',
     'other_folder': '3A1-3',
     'haddock_folder': '3A1-3/395662-GREM1_3A1-3_summary',
     'ab_chain': 'B', 'grem_chain': 'A', 'grem': 1,
     'sequenceh': 'QVQLQQPGAELVKPGASVKLSCKASGNTFTSYWMHWVKQRPGQGLEWIGMIHPNSGNTYYNEKFKSKTTLTVDKSSSTAYMQLSSLTSEDSAVYYCARSRGLYYGSLDYWGQGTTLT',
     'sequencel': 'DVVMTQTPLSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKSGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK'},
    {'ab': '3a13_grem2', 'pdb_folder': 'alphafold/templates/3a13_grem2/',
     'other_folder': '3A1-3',
     'haddock_folder': '3A1-3/395663-GREM2_3A1-3_summary',
     'ab_chain': 'B', 'grem_chain': 'A', 'grem': 2,
     'sequenceh': 'QVQLQQPGAELVKPGASVKLSCKASGNTFTSYWMHWVKQRPGQGLEWIGMIHPNSGNTYYNEKFKSKTTLTVDKSSSTAYMQLSSLTSEDSAVYYCARSRGLYYGSLDYWGQGTTLT',
     'sequencel': 'DVVMTQTPLSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKSGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK'},
    {'ab': '14d10_grem1', 'pdb_folder': 'alphafold/templates/14d10_grem1/',
     'other_folder': '14D10',
     'haddock_folder': '14D10/395665-GREM1_14D10_summary',
     'ab_chain': 'B', 'grem_chain': 'A', 'grem': 1,
     'sequenceh': 'EVQLQQSGPELVKPGASVKISCKASGYSFTGYYMHWVKQSHGNILDWIGYFFPYNGFSNCNQKFKGKATLTVDKSSSTAYMELRSLTSEDSAVYYCARGGLGRGYFDVWGTGTTVTVSS',
     'sequencel': 'DIQMTQSPSYLSVSLGGRVTITCKASDHINNWLAWYQQKPGNAPRLLISGATSLETGVPSRFSGSGSGKDYTLSITSLQTEDVATYYCQQYWSSPRTFGGGTKLEIK'},
    {'ab': '14d10_grem2', 'pdb_folder': 'alphafold/templates/14d10_grem2/',
     'other_folder': '14D10',
     'haddock_folder': '14D10/395672-GREM2_14D10_summary',
     'ab_chain': 'B', 'grem_chain': 'A', 'grem': 2,
     'sequenceh': 'EVQLQQSGPELVKPGASVKISCKASGYSFTGYYMHWVKQSHGNILDWIGYFFPYNGFSNCNQKFKGKATLTVDKSSSTAYMELRSLTSEDSAVYYCARGGLGRGYFDVWGTGTTVTVSS',
     'sequencel': 'DIQMTQSPSYLSVSLGGRVTITCKASDHINNWLAWYQQKPGNAPRLLISGATSLETGVPSRFSGSGSGKDYTLSITSLQTEDVATYYCQQYWSSPRTFGGGTKLEIK'},
]

P_THRESHOLD = 0.5

# ═══════════════════════════════════════════════════════════════════
# PSBDM PARAMETERS
# ═══════════════════════════════════════════════════════════════════
PSBDM_WEIGHT = 1.0  # Weight for MSA term in PSBDM score
PSBDM_PSEUDOCOUNT = 0.01  # Pseudocount for log calculation

# ═══════════════════════════════════════════════════════════════════
# BLOSUM62 lookup
# ═══════════════════════════════════════════════════════════════════
BLOSUM62 = scripts.BLOSUM62
BLOSUM_KEYS = list(BLOSUM62.keys())
STANDARD_AA = list('ACDEFGHIKLMNPQRSTVWY')

def get_blosum(aa1, aa2):
    return BLOSUM62[aa1][BLOSUM_KEYS.index(aa2)]

sigmoid_v = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))

# ═══════════════════════════════════════════════════════════════════
# PDB CLEANING & STRUCTURAL PROXIMITY (--clean only)
# ═══════════════════════════════════════════════════════════════════
def clean_and_analyze_haddock(cfg):
    """
    Split HADDOCK-docked PDBs into H/L/G chains, compute per-epitope
    residue interaction counts, save CSVs and scatter plots.
    """
    haddock_path = cfg['haddock_folder']
    if not os.path.isdir(haddock_path):
        print(f'  HADDOCK folder not found: {haddock_path}, skipping clean.')
        return
    grem = cfg['grem']
    ab_name = cfg['other_folder']
    epitope_residues = EPITOPE_GREM1 if grem == 1 else EPITOPE_GREM2

    csv_outdir = os.path.join(haddock_path, f'csvs_grem{grem}')
    os.makedirs(csv_outdir, exist_ok=True)
    plotpath = os.path.join(haddock_path, 'plots')
    os.makedirs(plotpath, exist_ok=True)

    # 1. Split chains
    heavy_fasta = f'{cfg["other_folder"]}/heavy.fasta'
    light_fasta = f'{cfg["other_folder"]}/light.fasta'
    print(f'  Splitting PDB chains in {haddock_path} ...')
    sumfile = scripts.split_chain_and_rename(haddock_path, heavy_fasta, light_fasta,
                                             cfg['ab_chain'], cfg['grem_chain'])
    if sumfile:
        total_count = sumfile
    else:
        total_count = len([f for f in os.listdir(haddock_path) if f.endswith('.pdb') and 'modified' not in f]) // 2

    # 2. Find modified PDB files
    pdb_files = [os.path.join(haddock_path, f) for f in os.listdir(haddock_path)
                 if 'modified' in f and f.endswith('.pdb')]
    if not pdb_files:
        print(f'  No modified PDBs found in {haddock_path}')
        return
    print(f'  Found {len(pdb_files)} modified PDB files.')

    # 3. Compute per-epitope residue interaction counts
    e = {f'{res_id}': [] for res_id in EPITOPE_IDS}
    for pdb_file in pdb_files:
        for res_id in EPITOPE_IDS:
            residue_id = (' ', res_id, ' ')
            nearby = scripts.find_residues_within_radius(pdb_file, residue_id, CHAIN_ID, radius=5.5)
            e[f'{res_id}'] += nearby

    # 4. Save per-epitope CSVs
    all_id_dicts = {}
    for res_id in EPITOPE_IDS:
        num_dict = dict(Counter(e[f'{res_id}']))
        idx = res_id - EPITOPE_IDS[0]
        label = f'{epitope_residues[idx]}_{res_id}'
        all_id_dicts[label] = num_dict
        scripts.save_probabilities_and_counts_to_csv(label, num_dict, csv_outdir, total_count)

    df_hl = scripts.save_heavy_light_probabilities_to_csv(all_id_dicts)
    df_hl.to_csv(os.path.join(csv_outdir, 'H_L_prob.csv'), sep='\t', index=False)

    # 5. Plot per-epitope scatter and overall binding probability
    dict_of_dfs = {}
    for fname in os.listdir(csv_outdir):
        if 'H_L_prob' in fname or 'epitope' in fname or 'Paratope' in fname:
            continue
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(csv_outdir, fname)
        base = fname.replace('.csv', '')
        parts = base.split('_')
        if len(parts) < 2:
            continue
        name_idx = int(parts[1]) - EPITOPE_IDS[0]
        plot_name = f'GREM{grem} {parts[0]}{name_idx}'
        df_plot = pd.read_csv(fpath, header=0, index_col=0)
        dict_of_dfs[plot_name] = df_plot
        outpath = os.path.join(plotpath, f'GREM{grem} {parts[0]}{name_idx}.png')
        scripts.scatter_plot_per_id(df_plot, plot_name, outpath)

    title = f'GREM{grem} Epitope Binding Probablity to {ab_name}'
    scripts.plot_epitope_binding_probablity(dict_of_dfs, title, csv_outdir)
    plt.close('all')

    # 6. Paratope probability plots
    para_dfs = []
    for fname in os.listdir(csv_outdir):
        if 'H_L_prob' in fname or 'epitope' in fname or 'Paratope' in fname:
            continue
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(csv_outdir, fname)
        df_p = pd.read_csv(fpath, header=0, index_col=0)
        if 'Probabilities' in df_p.index:
            df_p = df_p.drop('Probabilities')
        para_dfs.append(df_p)
    if para_dfs:
        para_concat = pd.concat(para_dfs).fillna(0.0)
        df_probs = pd.DataFrame(dict(np.clip(para_concat.sum() / total_count, 0, 1)), index=['Probs'])
        df_probs = df_probs.sort_values(by='Probs', axis=1, ascending=False)
        df_probs.to_csv(os.path.join(csv_outdir, 'Paratope_probabilities.csv'), index=False)
    plt.close('all')
    print(f'  Structural proximity analysis complete for {cfg["ab"]}.')


# ═══════════════════════════════════════════════════════════════════
# EPITOPE LOGO PLOTS (always run if CSVs exist)
# ═══════════════════════════════════════════════════════════════════
def plot_epitope_logo(cfg):
    """
    Read the epitope_binding_probablity.csv from the HADDOCK summary
    csvs folder and generate a sequence logo plot.
    """
    try:
        import logomaker
    except ImportError:
        print('  logomaker not installed, skipping logo plot. Install with: pip install logomaker')
        return

    haddock_path = cfg['haddock_folder']
    grem = cfg['grem']
    ab_name = cfg['other_folder']
    epitope_residues = EPITOPE_GREM1 if grem == 1 else EPITOPE_GREM2

    csv_path = os.path.join(haddock_path, f'csvs_grem{grem}', 'epitope_binding_probablity.csv')
    if not os.path.isfile(csv_path):
        print(f'  CSV not found: {csv_path}, skipping logo. Run with --clean first.')
        return

    os.makedirs('seqlogo', exist_ok=True)

    # Read and parse the CSV
    df_raw = pd.read_csv(csv_path)

    # Convert column names: "GREM2 E29" -> "GREM2_E_29" style
    col_map = {}
    for col in df_raw.columns:
        parts = col.strip().split(' ')
        if len(parts) >= 2:
            grem_part = parts[0]
            rest = parts[1]
            aa_letter = rest[0]
            num = rest[1:]
            col_map[col] = f'{grem_part}_{aa_letter}_{num}'
        else:
            col_map[col] = col
    df_raw = df_raw.rename(columns=col_map)

    # Count modified PDBs to get num_structures
    modified_pdbs = [f for f in os.listdir(haddock_path) if 'modified' in f and f.endswith('.pdb')]
    num_structures = len(modified_pdbs) if modified_pdbs else 40  # fallback

    # Build interaction frequency matrix
    epi_dict = {}
    for aa in epitope_residues:
        epi_dict[aa] = [0.0] * len(epitope_residues)

    for i, aa in enumerate(epitope_residues):
        col_name = f'GREM{grem}_{aa}_{i}'
        if col_name in df_raw.columns:
            epi_dict[aa][i] = float(df_raw[col_name].iloc[0]) / num_structures

    df_logo = pd.DataFrame(epi_dict)

    # =========================
    # SAVE MATRIX AS CSV (NEW)
    # =========================
    csv_out_path = f'seqlogo/{ab_name}_GREM{grem}_logo_matrix.csv'
    df_logo.to_csv(csv_out_path, index=True)
    print(f'  Saved: {csv_out_path}')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 3))
    logo = logomaker.Logo(df_logo, color_scheme='NajafabadiEtAl2017', ax=ax)
    logo.ax.set_ylabel('Relative Interaction Frequency')
    logo.ax.set_xlabel('Sequence')
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.style_xticks(anchor=0, spacing=1, rotation=90)

    name = f'{ab_name} GREM{grem}'
    plt.tight_layout()

    png_path = f'seqlogo/{name}.png'
    pdf_path = f'seqlogo/{name}.pdf'

    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f'  Saved: {png_path}')
    print(f'  Saved: {pdf_path}')

# ═══════════════════════════════════════════════════════════════════
# STAGE 1 — Position Selection
# ═══════════════════════════════════════════════════════════════════
def stage1_position_selection(proabc2_df, dist_scores_chain, conservation_df, sequence, frequency_df, p_thr=P_THRESHOLD):
    P_i = np.asarray(scripts.calculate_paratope_probability_score(proabc2_df), dtype=np.float64)
    D_i = np.asarray(dist_scores_chain, dtype=np.float64)
    C_i = np.asarray(scripts.calculate_conservation_score(conservation_df), dtype=np.float64)
    F_i = np.array([np.exp(-15.0 * float(frequency_df.loc[aa, f'Pos_{pos+1}']))
                     for pos, aa in enumerate(sequence)], dtype=np.float64)
    R_i = P_i * D_i * C_i * F_i
    df = pd.DataFrame({
        'P_i': P_i, 'D_i': D_i, 'C_i': C_i, 'F_i': F_i, 'R_i': R_i,
        'sequence': list(sequence),
    }, index=[f'Pos_{i}' for i in range(1, len(sequence) + 1)])
    df['selected'] = df['P_i'] > p_thr
    df['rank'] = df.loc[df['selected'], 'R_i'].rank(ascending=False, method='min')
    return df


# ═══════════════════════════════════════════════════════════════════
# STAGE 2 — Mutation Ranking via PSBDM (Position-Specific BLOSUM Dissimilarity Matrix)
# ═══════════════════════════════════════════════════════════════════
def stage2_psbdm_matrix(sequence, msa_freq_df, weight=PSBDM_WEIGHT, pseudocount=PSBDM_PSEUDOCOUNT):
    """
    Compute Position-Specific BLOSUM Dissimilarity Matrix (PSBDM).
    
    Uses the new formula: Score(j) = B(i, j) + [w * P(k, j)]
    Where:
    - B(i, j) = BLOSUM62 score between wild-type i and mutant j
    - w = weight parameter
    - P(k, j) = 2 * log2( (f_k(j) + pseudocount) / q_j )
    
    LOWER scores indicate BETTER mutations (more dissimilar + rare at that position).
    """
    return scripts.compute_psbdm_matrix(sequence, msa_freq_df, weight=weight, pseudocount=pseudocount)


def stage2_blosum_matrix_legacy(sequence):
    """
    Legacy BLOSUM62 dissimilarity matrix for comparison.
    Uses: score(a,b) = BLOSUM62(a,a) - BLOSUM62(a,b)
    """
    score_dict = {}
    for pos, aa_wt in enumerate(sequence):
        col = f'Pos_{pos + 1}'
        self_score = get_blosum(aa_wt, aa_wt)
        scores = []
        for aa in STANDARD_AA:
            if aa == aa_wt:
                scores.append(np.nan)
            else:
                scores.append(self_score - get_blosum(aa_wt, aa))
        score_dict[col] = scores
    return pd.DataFrame(score_dict, index=STANDARD_AA)


def build_mutation_summary(stage1_df, psbdm_df, chain_label):
    """
    Build mutation summary from PSBDM scores.
    LOWER PSBDM scores = BETTER mutations (more dissimilar + rare).
    """
    rows = []
    for pos_label in stage1_df[stage1_df['selected']].sort_values('R_i', ascending=False).index:
        if pos_label not in psbdm_df.columns:
            continue
        scores = psbdm_df[pos_label].dropna()
        if len(scores) == 0:
            continue
        # For PSBDM, lower is better, so use nsmallest
        top5 = scores.nsmallest(5)
        rows.append({
            'position': pos_label, 'chain': chain_label,
            'wt_aa': stage1_df.loc[pos_label, 'sequence'],
            'P_i': round(stage1_df.loc[pos_label, 'P_i'], 4),
            'D_i': round(stage1_df.loc[pos_label, 'D_i'], 4),
            'C_i': round(stage1_df.loc[pos_label, 'C_i'], 4),
            'R_i': round(stage1_df.loc[pos_label, 'R_i'], 4),
            'best_mutation': top5.index[0],
            'psbdm_score': round(top5.values[0], 3),
            'top5_mutations': ','.join(top5.index),
            'top5_scores': ','.join(f'{v:.3f}' for v in top5.values),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════
def plot_score_barplot(score_gr1, score_gr2, ab_prefix, chain, outpath):
    score = pd.DataFrame({
        'GREM1': score_gr1['R_i'].values,
        'GREM2': score_gr2['R_i'].values,
        'sequence': score_gr1['sequence'].values,
    })
    # Save data to CSV
    csv_path = outpath.replace('.png', '_data.csv')
    score_export = score.copy()
    score_export['position'] = [f'Pos_{i+1}' for i in range(len(score))]
    score_export.to_csv(csv_path, index=False)
    
    x = np.arange(len(score))
    fig, ax = plt.subplots(figsize=(max(12, len(score) * 0.15), 5))
    w = 0.35
    ax.bar(x - w/2, score['GREM1'], w, label='GREM1', color='steelblue', edgecolor='black', linewidth=0.3)
    ax.bar(x + w/2, score['GREM2'], w, label='GREM2', color='coral', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Position', fontsize=11)
    ax.set_ylabel('$R_i = P_i \\cdot D_i \\cdot C_i \\cdot F_i$', fontsize=11)
    ax.set_title(f'{ab_prefix} — {chain.capitalize()} Chain', fontsize=13)
    tick_pos = list(range(0, len(score), 5))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f'{score["sequence"].iloc[i]}{i+1}' for i in tick_pos], rotation=90, fontsize=7)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_scatter(score_gr1, score_gr2, ab_prefix, chain, outpath):
    x = np.arange(1, len(score_gr1) + 1)
    # Save data to CSV
    csv_path = outpath.replace('.png', '_data.csv')
    score_export = pd.DataFrame({
        'position': [f'Pos_{i}' for i in x],
        'GREM1_R_i': score_gr1['R_i'].values,
        'GREM2_R_i': score_gr2['R_i'].values,
        'sequence': score_gr1['sequence'].values,
    })
    score_export.to_csv(csv_path, index=False)
    
    fig, ax = plt.subplots(figsize=(max(10, len(score_gr1) * 0.12), 4))
    ax.plot(x, score_gr1['R_i'].values, color='steelblue', label='GREM1', alpha=0.7)
    ax.plot(x, score_gr2['R_i'].values, color='coral', label='GREM2', alpha=0.7)
    ax.set_xlabel('Position', fontsize=11)
    ax.set_ylabel('$R_i$', fontsize=11)
    ax.set_title(f'{ab_prefix} — {chain.capitalize()} Chain Position Profile', fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_sequence_profile(stage1_df, chain_label, outpath):
    # Save data to CSV
    csv_path = outpath.replace('.png', '_data.csv')
    stage1_df.to_csv(csv_path)
    
    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(stage1_df) * 0.14), 6), sharex=True)
    x = range(len(stage1_df))
    sel = stage1_df['selected']
    axes[0].bar(x, stage1_df['P_i'], color=['coral' if s else 'lightgrey' for s in sel], width=1.0)
    axes[0].axhline(P_THRESHOLD, color='black', linestyle='--', linewidth=1, label=f'threshold={P_THRESHOLD}')
    axes[0].set_ylabel('$P_i$')
    axes[0].legend(fontsize=8)
    axes[0].set_title(f'{chain_label} Chain — Stage 1 Profile')
    axes[1].bar(x, stage1_df['R_i'], color=['steelblue' if s else 'lightgrey' for s in sel], width=1.0)
    axes[1].set_ylabel('$R_i = P_i \\cdot D_i \\cdot C_i \\cdot F_i$')
    axes[1].set_xlabel('Residue Position')
    top10 = stage1_df[sel].nlargest(10, 'R_i')
    for idx in top10.index:
        pn = int(idx.split('_')[1]) - 1
        axes[1].annotate(f"{stage1_df.loc[idx, 'sequence']}{idx.split('_')[1]}",
                         xy=(pn, stage1_df.loc[idx, 'R_i']),
                         xytext=(0, 8), textcoords='offset points',
                         fontsize=6, ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_psbdm_heatmap(psbdm_df, stage1_df, chain_label, outpath, topn=10):
    """
    Plot PSBDM heatmap. Lower values = better mutations (shown in green).
    """
    top = stage1_df[stage1_df['selected']].sort_values('R_i', ascending=False).head(topn).index.tolist()
    sub = psbdm_df[[p for p in top if p in psbdm_df.columns]].copy()
    rename = {p: f"{p} ({stage1_df.loc[p, 'sequence']})" for p in sub.columns}
    sub = sub.rename(columns=rename)
    
    # Save heatmap data to CSV
    csv_path = outpath.replace('.png', '_data.csv')
    sub.to_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(max(8, len(sub.columns) * 1.2), 8))
    # Reverse colormap: lower (better) = green, higher = red
    sns.heatmap(sub, cmap='RdYlGn_r', center=sub.values[~np.isnan(sub.values)].mean(), 
                annot=True, fmt='.1f', linewidths=0.5, ax=ax,
                cbar_kws={'label': 'PSBDM Score (lower = better mutation)'})
    ax.set_title(f'Position-Specific BLOSUM Dissimilarity — {chain_label} Chain (top {topn})', fontsize=12)
    ax.set_ylabel('Candidate AA')
    ax.set_xlabel('Position (WT)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# FIGURE — Mutation Heatmap for HuVar7 (specific positions)
# ═══════════════════════════════════════════════════════════════════
def plot_mutation_heatmap_huvar7(all_results, outdir='results/figures'):
    os.makedirs(outdir, exist_ok=True)
    huvar7_key = None
    for k in all_results:
        if k.startswith('huvar7'):
            huvar7_key = k
            break
    if huvar7_key is None:
        print('  HuVar7 results not found, skipping mutation heatmap.')
        return
    res = all_results[huvar7_key]
    positions = [
        ('G_31', 'heavy', 'Pos_31'),
        ('F_52', 'heavy', 'Pos_52'),
        ('F_57', 'heavy', 'Pos_57'),
        ('L_101', 'heavy', 'Pos_101'),
        ('W_92', 'light', 'Pos_92'),
    ]
    aa_order = list('ACDEFGHIKLMNPQRSTVWY')
    data, col_labels, wt_aas, chain_labels = [], [], [], []
    for label, chain, pos_key in positions:
        psbdm_df = res[chain]['psbdm']
        stage1_df = res[chain]['stage1']
        if pos_key not in psbdm_df.columns:
            continue
        col = psbdm_df[pos_key].reindex(aa_order).values.astype(float)
        data.append(col)
        col_labels.append(label)
        wt_aas.append(stage1_df.loc[pos_key, 'sequence'])
        chain_labels.append(chain)
    if not data:
        print('  No data for HuVar7 mutation heatmap.')
        return
    mat = np.column_stack(data)
    df_raw = pd.DataFrame(mat, index=aa_order, columns=col_labels)
    
    # Save heatmap data to CSV
    df_raw.to_csv(f'{outdir}/huvar7_mutation_heatmap_data.csv')
    
    mask = np.zeros(mat.shape, dtype=bool)
    for j, wt in enumerate(wt_aas):
        for i, aa in enumerate(aa_order):
            if aa == wt:
                mask[i, j] = True
    n_heavy = sum(1 for c in chain_labels if c == 'heavy')
    for cmap_name, cmap, suffix in [('GnRd', 'RdYlGn_r', 'green_red'), ('BlRd', 'RdBu_r', 'blue_red')]:
        fig, ax = plt.subplots(figsize=(max(4, len(col_labels) * 1.1), 7))
        sns.heatmap(df_raw, cmap=cmap, linewidths=0.8, linecolor='white', ax=ax,
                    annot=False, cbar_kws={'label': 'PSBDM Score (lower = better)', 'shrink': 0.7}, mask=mask)
        for j, wt in enumerate(wt_aas):
            for i, aa in enumerate(aa_order):
                if aa == wt:
                    ax.plot([j, j+1], [i, i+1], color='black', linewidth=1.5, clip_on=True)
                    ax.plot([j, j+1], [i+1, i], color='black', linewidth=1.5, clip_on=True)
        if n_heavy > 0:
            ax.text(n_heavy / 2, -0.8, 'heavy chain', ha='center', va='bottom', fontsize=10, fontweight='bold')
        if n_heavy < len(col_labels):
            ax.text(n_heavy + (len(col_labels) - n_heavy) / 2, -0.8, 'light chain',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        if 0 < n_heavy < len(col_labels):
            ax.axvline(x=n_heavy, color='black', linewidth=2)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=9)
        plt.tight_layout()
        plt.savefig(f'{outdir}/huvar7_mutation_heatmap_{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {outdir}/huvar7_mutation_heatmap_{suffix}.png')


# ═══════════════════════════════════════════════════════════════════
# FIGURE — Paratoping Score Bar Plot (GREM1 black / GREM2 white)
# ═══════════════════════════════════════════════════════════════════
def plot_paratoping_barplot(all_results, outdir='results/figures'):
    os.makedirs(outdir, exist_ok=True)
    prefixes = {}
    for ab_name, res in all_results.items():
        prefix = ab_name.split('_')[0]
        grem = 'grem1' if 'grem1' in ab_name else 'grem2'
        prefixes.setdefault(prefix, {})[grem] = res
    for prefix, grem_dict in prefixes.items():
        if 'grem1' not in grem_dict or 'grem2' not in grem_dict:
            continue
        for chain in ['heavy', 'light']:
            s1_gr1 = grem_dict['grem1'][chain]['stage1']
            s1_gr2 = grem_dict['grem2'][chain]['stage1']
            sel_positions = sorted(set(
                s1_gr1[s1_gr1['selected']].index.tolist() +
                s1_gr2[s1_gr2['selected']].index.tolist()
            ), key=lambda p: int(p.split('_')[1]))
            if not sel_positions:
                continue
            labels, vals1, vals2 = [], [], []
            for pos in sel_positions:
                pos_num = pos.split('_')[1]
                aa = s1_gr1.loc[pos, 'sequence'] if pos in s1_gr1.index else s1_gr2.loc[pos, 'sequence']
                labels.append(f'{pos_num}_{aa}')
                vals1.append(float(s1_gr1.loc[pos, 'R_i']) if pos in s1_gr1.index else 0.0)
                vals2.append(float(s1_gr2.loc[pos, 'R_i']) if pos in s1_gr2.index else 0.0)
            sort_idx = np.argsort([-max(v1, v2) for v1, v2 in zip(vals1, vals2)])
            labels = [labels[i] for i in sort_idx]
            vals1 = [vals1[i] for i in sort_idx]
            vals2 = [vals2[i] for i in sort_idx]
            # Keep only top 10
            labels, vals1, vals2 = labels[:10], vals1[:10], vals2[:10]
            
            # Save data to CSV
            csv_path = f'{outdir}/{prefix}_paratoping_{chain}_data.csv'
            pd.DataFrame({
                'position': labels,
                'GREM1_R_i': vals1,
                'GREM2_R_i': vals2
            }).to_csv(csv_path, index=False)
            
            x = np.arange(len(labels))
            w = 0.35
            fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.45), 4))
            ax.bar(x - w/2, vals1, w, label='GREM1', color='black', edgecolor='black', linewidth=0.5)
            ax.bar(x + w/2, vals2, w, label='GREM2', color='white', edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_ylabel('Paratoping scores', fontsize=11)
            ax.set_title(f'Predicted paratope residues on {chain} chain', fontsize=12)
            ax.legend(fontsize=9, frameon=True)
            ax.set_xlim(-0.5, len(labels) - 0.5)
            plt.tight_layout()
            plt.savefig(f'{outdir}/{prefix}_paratoping_{chain}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f'  Saved: {outdir}/{prefix}_paratoping_{chain}.png')


# ═══════════════════════════════════════════════════════════════════
# SAVE R_i SCORES CSV
# ═══════════════════════════════════════════════════════════════════
def save_ri_scores_csv(all_results, outdir='results'):
    """Save R_i scores for all antibodies as CSV files."""
    os.makedirs(outdir, exist_ok=True)
    
    for ab_name, res in all_results.items():
        for chain in ['heavy', 'light']:
            stage1_df = res[chain]['stage1']
            ri_df = pd.DataFrame({
                'position': stage1_df.index,
                'sequence': stage1_df['sequence'],
                'P_i': stage1_df['P_i'],
                'D_i': stage1_df['D_i'],
                'C_i': stage1_df['C_i'],
                'F_i': stage1_df['F_i'],
                'R_i': stage1_df['R_i'],
                'selected': stage1_df['selected']
            })
            csv_path = f'{outdir}/{ab_name}_{chain}_Ri_scores.csv'
            ri_df.to_csv(csv_path, index=False)
            print(f'  Saved R_i scores: {csv_path}')


# ═══════════════════════════════════════════════════════════════════
# SAVE HUVAR7 FULL PSBDM AND P(k,j) MATRICES
# ═══════════════════════════════════════════════════════════════════
def save_huvar7_full_matrices(all_results, outdir='results/huvar7_matrices'):
    """
    Save full PSBDM matrices and raw P(k,j) matrices for HuVar7.
    Rows = 20 amino acids, Columns = positions.
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Find HuVar7 results
    huvar7_keys = [k for k in all_results if k.startswith('huvar7')]
    if not huvar7_keys:
        print('  HuVar7 results not found, skipping full matrix export.')
        return
    
    # Use first HuVar7 result (grem1 or grem2 - matrices are the same for same antibody)
    huvar7_key = huvar7_keys[0]
    res = all_results[huvar7_key]
    
    for chain in ['heavy', 'light']:
        psbdm_df = res[chain]['psbdm']
        p_scores_df = res[chain]['p_scores']
        
        # Save PSBDM matrix (20 AA rows x positions columns)
        psbdm_path = f'{outdir}/huvar7_{chain}_PSBDM_full.csv'
        psbdm_df.to_csv(psbdm_path)
        print(f'  Saved full PSBDM matrix: {psbdm_path}')
        
        # Save P(k,j) matrix (20 AA rows x positions columns)
        pkj_path = f'{outdir}/huvar7_{chain}_Pkj_raw.csv'
        p_scores_df.to_csv(pkj_path)
        print(f'  Saved raw P(k,j) matrix: {pkj_path}')


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════
def run_single_ab(cfg):
    ab = cfg['ab']
    a = ab.split('_')[0]
    print(f'\n{"="*60}\n  Processing: {ab}\n{"="*60}')
    outdir = f'results/{ab}'
    os.makedirs(outdir, exist_ok=True)
    
    # Load MSA data
    freq_h = scripts.compute_aa_frequencies(f'antibody_sequences/{a}/aln_heavy_{a}.fa')
    freq_l = scripts.compute_aa_frequencies(f'antibody_sequences/{a}/aln_light_{a}.fa')
    cons_h = scripts.compute_conservation_percentage(f'antibody_sequences/{a}/aln_heavy_{a}.fa')
    cons_l = scripts.compute_conservation_percentage(f'antibody_sequences/{a}/aln_light_{a}.fa')
    
    # Load ProABC-2 predictions
    prob_h = pd.read_csv(f'{cfg["other_folder"]}/heavy-pred.csv').transpose()
    prob_h.columns = [f'Pos_{i}' for i in range(1, len(prob_h.columns) + 1)]
    prob_l = pd.read_csv(f'{cfg["other_folder"]}/light-pred.csv').transpose()
    prob_l.columns = [f'Pos_{i}' for i in range(1, len(prob_l.columns) + 1)]
    
    # Load distance scores
    dist_hl, _ = scripts.calculate_distance_score_from_folder(cfg['pdb_folder'])
    print(f'  PDB files: {len([f for f in os.listdir(cfg["pdb_folder"]) if f.endswith(".pdb")])}')
    
    results = {}
    for chain_label, seq, prob_df, dist_arr, cons_df, freq_df in [
        ('heavy', cfg['sequenceh'], prob_h, dist_hl['H'], cons_h, freq_h),
        ('light', cfg['sequencel'], prob_l, dist_hl['L'], cons_l, freq_l),
    ]:
        # Stage 1: Position Selection
        s1 = stage1_position_selection(prob_df, dist_arr, cons_df, seq, freq_df)
        n_sel = s1['selected'].sum()
        print(f'  {chain_label.capitalize()} chain: {n_sel} positions selected (P_i > {P_THRESHOLD})')
        
        # Stage 2: PSBDM Matrix
        psbdm_mat, p_scores = stage2_psbdm_matrix(seq, freq_df, weight=PSBDM_WEIGHT, pseudocount=PSBDM_PSEUDOCOUNT)
        
        # Build mutation summary
        summary = build_mutation_summary(s1, psbdm_mat, chain_label[0].upper())
        
        # Save results
        s1.to_csv(f'{outdir}/{ab}_{chain_label}chain_stage1.csv')
        psbdm_mat.to_csv(f'{outdir}/{ab}_{chain_label}chain_PSBDM.csv')
        p_scores.to_csv(f'{outdir}/{ab}_{chain_label}chain_Pkj.csv')
        summary.to_csv(f'{outdir}/{ab}_{chain_label}chain_mutations.csv', index=False)
        
        # Plots
        plot_sequence_profile(s1, f'{ab} {chain_label.capitalize()}', f'{outdir}/{ab}_{chain_label}_profile.png')
        plot_psbdm_heatmap(psbdm_mat, s1, f'{ab} {chain_label.capitalize()}', f'{outdir}/{ab}_{chain_label}_psbdm_heatmap.png')
        
        results[chain_label] = {'stage1': s1, 'psbdm': psbdm_mat, 'p_scores': p_scores, 'summary': summary}
        
        print(f'  Top 5 {chain_label} positions (lower PSBDM = better mutation):')
        for _, row in summary.head(5).iterrows():
            print(f"    {row['position']} ({row['wt_aa']}) → {row['best_mutation']}  "
                  f"R_i={row['R_i']:.3f}  PSBDM={row['psbdm_score']:.2f}")
    
    return results


def make_comparison_plots(all_results):
    figdir = 'results/figures'
    os.makedirs(figdir, exist_ok=True)
    prefixes = {}
    for ab_name, res in all_results.items():
        prefix = ab_name.split('_')[0]
        grem = 'grem1' if 'grem1' in ab_name else 'grem2'
        prefixes.setdefault(prefix, {})[grem] = res
    for prefix, grem_dict in prefixes.items():
        if 'grem1' not in grem_dict or 'grem2' not in grem_dict:
            continue
        for chain in ['heavy', 'light']:
            s1_gr1 = grem_dict['grem1'][chain]['stage1']
            s1_gr2 = grem_dict['grem2'][chain]['stage1']
            n = min(len(s1_gr1), len(s1_gr2))
            plot_score_barplot(s1_gr1.iloc[:n], s1_gr2.iloc[:n], prefix, chain, f'{figdir}/{prefix}_grem_{chain}.png')
            plot_score_scatter(s1_gr1.iloc[:n], s1_gr2.iloc[:n], prefix, chain, f'{figdir}/{prefix}_grem_{chain}_scatter.png')
            print(f'  Saved: {figdir}/{prefix}_grem_{chain}.png')


def write_global_summary(all_results):
    summary = {}
    lines = ['=' * 70, 'GLOBAL SUMMARY — Two-Stage Paratope Scoring with PSBDM', '=' * 70,
             f'P_i threshold: {P_THRESHOLD}', 
             f'Stage 1 ranking: R_i = P_i * D_i * C_i * exp(-15·Freq_wt)',
             f'Stage 2 mutations: PSBDM Score = B(wt, mut) + w * P(k, mut)',
             f'  where P(k, j) = 2 * log2( (f_k(j) + pseudocount) / q_j )',
             f'  w = {PSBDM_WEIGHT}, pseudocount = {PSBDM_PSEUDOCOUNT}',
             f'  Lower PSBDM = better mutation (more dissimilar + rare at position)', '']
    for ab_name, res in all_results.items():
        lines.append(f'\n--- {ab_name} ---')
        ab_summary = {}
        for chain in ['heavy', 'light']:
            s = res[chain]['summary']
            n_sel = len(s)
            lines.append(f'  {chain.capitalize()}: {n_sel} selected positions')
            if len(s) > 0:
                for _, row in s.head(5).iterrows():
                    lines.append(f"    {row['position']} ({row['wt_aa']}) → {row['best_mutation']}  "
                                f"R_i={row['R_i']:.3f}  PSBDM={row['psbdm_score']:.2f}")
            ab_summary[chain] = {
                'n_selected': n_sel,
                'top5': s.head(5)[['position', 'wt_aa', 'best_mutation', 'R_i', 'psbdm_score']].to_dict('records') if len(s) > 0 else []
            }
        summary[ab_name] = ab_summary
    lines.append('\n' + '=' * 70)
    report_text = '\n'.join(lines)
    os.makedirs('results', exist_ok=True)
    with open('results/SUMMARY_REPORT.txt', 'w') as f:
        f.write(report_text)
    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\n{report_text}')


# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-Stage Paratope Scoring Pipeline with PSBDM')
    parser.add_argument('--clean', action='store_true',
                        help='Re-run PDB cleaning: split HADDOCK chains, compute structural '
                             'proximity CSVs, and generate per-epitope plots.')
    parser.add_argument('--weight', type=float, default=1.0,
                        help='Weight parameter for MSA term in PSBDM (default: 1.0)')
    parser.add_argument('--pseudocount', type=float, default=0.01,
                        help='Pseudocount for log calculation in P(k,j) (default: 0.01)')
    args = parser.parse_args()
    
    # Update global parameters if provided
    PSBDM_WEIGHT = args.weight
    PSBDM_PSEUDOCOUNT = args.pseudocount

    # Step 0 (optional): Clean PDBs and structural proximity analysis
    if args.clean:
        print('\n' + '=' * 60)
        print('  STEP 0: PDB Cleaning & Structural Proximity Analysis')
        print('=' * 60)
        for cfg in ANTIBODIES:
            try:
                clean_and_analyze_haddock(cfg)
            except Exception as e:
                print(f'  ERROR cleaning {cfg["ab"]}: {e}')
                import traceback; traceback.print_exc()

    # Step 1: Epitope logo plots (always, if CSVs exist)
    print('\n' + '=' * 60)
    print('  Epitope Logo Plots')
    print('=' * 60)
    for cfg in ANTIBODIES:
        try:
            plot_epitope_logo(cfg)
        except Exception as e:
            print(f'  ERROR logo for {cfg["ab"]}: {e}')

    # Step 2: Scoring pipeline
    all_results = {}
    for cfg in ANTIBODIES:
        try:
            res = run_single_ab(cfg)
            all_results[cfg['ab']] = res
        except Exception as e:
            print(f'  ERROR processing {cfg["ab"]}: {e}')
            import traceback; traceback.print_exc()

    # Step 3: Figures
    make_comparison_plots(all_results)
    plot_mutation_heatmap_huvar7(all_results)
    plot_paratoping_barplot(all_results)
    
    # Step 4: Save R_i scores CSV
    save_ri_scores_csv(all_results)
    
    # Step 5: Save full PSBDM and P(k,j) matrices for HuVar7
    save_huvar7_full_matrices(all_results)
    
    write_global_summary(all_results)
    print('\nDone. All results in results/')
