import os
from Bio.PDB import PDBParser, PDBIO, Chain, Model, NeighborSearch
from Bio import SeqIO, PDB
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.StructureBuilder import StructureBuilder
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import scipy
import math

def renumber_pdb_residues(pdb_filename, output_filename):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_filename)

    residue_counter = 1

    for model in structure:
        for chain in model:
            for residue in chain:
                residue.id = (' ', residue_counter, ' ')
                residue_counter += 1

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_filename)

def split_chain_and_rename(pdb_directory, heavy_fasta, light_fasta, AB='A', Grem='B'):
    # Read the heavy and light sequences from FASTA files
    with open(heavy_fasta, "r") as heavy_file:
        heavy_seq = str(next(SeqIO.parse(heavy_file, "fasta")).seq)
    
    with open(light_fasta, "r") as light_file:
        light_seq = str(next(SeqIO.parse(light_file, "fasta")).seq)
    
    # Initialize PDB parser and writer
    parser = PDBParser(QUIET=True)
    pdb_io = PDBIO()
    sumfile = 0
    # Iterate over all PDB files in the directory
    for pdb_file in os.listdir(pdb_directory):
        if pdb_file.endswith(".pdb"):
            pdb_path = os.path.join(pdb_directory, pdb_file)
            structure = parser.get_structure(pdb_file, pdb_path)

            # Get model (assuming only one model, index 0)
            model = structure[0]

            # Get chain A and B
            chain_a = model[AB]
            chain_b = model[Grem]
            
            # Create new chains for heavy (H), light (L), and rename B to G
            heavy_chain = Chain.Chain('H')
            light_chain = Chain.Chain('L')
            chain_b.id = 'G'  # Rename chain B to G

            index = 0


            for res in chain_a.get_residues():
                res_name = res.resname
                if index < len(heavy_seq):
                    heavy_chain.add(res)
                    index += 1
                else:
                    light_chain.add(res)
                    index += 1

            model.detach_child(AB)
            model.detach_child('G')
            model.add(chain_b)
            model.add(heavy_chain)
            model.add(light_chain)

            # Save the modified structure back to a new file
            output_path = os.path.join(pdb_directory, f"modified_{pdb_file}")
            pdb_io.set_structure(structure)
            pdb_io.save(output_path)
            
            renumber_pdb_residues(output_path, output_path)
            sumfile = sumfile + 1

            print(f"Processed and saved {output_path}")
    return sumfile

def find_residues_within_radius(pdb_file, residue_id, chain_id, radius=5.0):
    """
    Find and list amino acids within a specified radius around the C-alpha atom 
    of a given residue (residue_id) in a specific chain, but from other chains.
    
    Args:
    - pdb_file (str): Path to the PDB file.
    - residue_id (tuple): The residue ID in the format (residue_number, insertion_code).
                          For example, (50, ' ') where ' ' indicates no insertion code.
    - chain_id (str): Chain ID where the residue is located.
    - radius (float): The radius to search around the C-alpha atom. Default is 5.5Å.
    
    Returns:
    - List of residues from other chains within the specified radius.
    """
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    # Get the model (assuming only one model, index 0)
    model = structure[0]
    
    # Find the specified chain
    chain = model[chain_id]

    # Find the specific residue by ID
    target_residue = chain[residue_id]


    # Get the C-alpha atom of the specified residue
    if 'CA' not in target_residue:
        raise ValueError(f"Residue {residue_id} in chain {chain_id} does not have a C-alpha atom.")
    
    ca_atom = target_residue['CA']

    # Get all atoms in the structure and use NeighborSearch to find atoms within the radius
    atoms = [atom for atom in structure.get_atoms() if is_aa(atom.get_parent(), standard=True)]
    ns = NeighborSearch(atoms)
    nearby_atoms = ns.search(ca_atom.coord, radius)

    # Filter for residues from other chains
    other_chain_residues = set()
    for atom in nearby_atoms:
        residue = atom.get_parent()
        chain = residue.get_parent()
        
        # Check if the residue is from a different chain
        if chain.id != chain_id and is_aa(residue):
            other_chain_residues.add(f'{chain.id}_{residue.id[1]}_{residue.resname}')
            

    # Return the list of residues from other chains
    return list(other_chain_residues)


def save_probabilities_and_counts_to_csv(id_name, id_dict, outdir, total_count):
    """
    Calculate the probabilities for each key in the dictionary and save both counts and probabilities 
    as a CSV file using pandas.
    Args:
    - id_name (str): The name to be used for the CSV file.
    - id_dict (dict): A dictionary with keys representing IDs and values representing their counts.
    Returns:
    - None
    """
    # Calculate the total count to normalize and get probabilities
    #total_count = sum(id_dict.values())

    # Calculate probabilities
    probabilities = {key: count / total_count for key, count in id_dict.items()}
    # Create a DataFrame from the counts and probabilities
    df = pd.DataFrame([id_dict, probabilities], index=['Counts', 'Probabilities'])
    # Write the DataFrame to a CSV file
    csv_filename = os.path.join(outdir, f"{id_name}.csv")
    df.to_csv(csv_filename)



def calculate_heavy_light_probabilities(id_dict):
    """
    Calculate the probabilities for 'Heavy' (H) and 'Light' (L) chains in the given dictionary.
    
    Args:
    - id_dict (dict): A dictionary with keys representing IDs and values representing their counts.
    
    Returns:
    - A dictionary with 'Heavy' and 'Light' probabilities.
    """
    total_count = sum(id_dict.values())
    heavy_count = sum(count for key, count in id_dict.items() if key.startswith('H'))
    light_count = sum(count for key, count in id_dict.items() if key.startswith('L'))
    
    # Calculate the probabilities
    if total_count == 0:
        heavy_prob = 0
        light_prob = 0
    else:
        heavy_prob = heavy_count / total_count
        light_prob = light_count / total_count
    
    return {'Heavy': heavy_prob, 'Light': light_prob}

def save_heavy_light_probabilities_to_csv(all_id_dicts):
    """
    Calculate and save heavy and light chain probabilities for multiple id_dicts into a CSV file.
    
    Args:
    - all_id_dicts (dict): A dictionary where each key is an id_i name and each value is a dictionary 
                           containing the counts for that id_i.
    
    Returns:
    - None
    """
    # Initialize a list to store probabilities for each id_i
    data = []
    
    # Iterate over each id_i and its dictionary
    for id_name, id_dict in all_id_dicts.items():
        probs = calculate_heavy_light_probabilities(id_dict)
        data.append({'id_i': id_name, 'Heavy': probs['Heavy'], 'Light': probs['Light']})
    
    # Create a DataFrame with the probabilities for each id_i
    df = pd.DataFrame(data)
    return df


def scatter_plot_per_id(df, name, outpath):
    # Step 1: Sort columns based on 'Probabilities' row values in descending order
    df_sorted = df.sort_values(by='Probabilities', axis=1, ascending=False)
    # Step 2: Plot a line plot
    plt.figure(figsize=(10, 7))
    plt.plot(df_sorted.columns, df_sorted.loc['Probabilities'], marker='o', color='black')
    # Step 3: Customize plot
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.xlabel("Ab Amino Acids", size=12)
    plt.ylabel("Probability of Binding", size=12)
    plt.title(name, size=14)
    # Display the plot
    plt.tight_layout()
    plt.savefig(outpath, dpi=600)
    plt.close()

def plot_epitope_binding_probablity(dict_of_dfs, name, outpath):
    sum_dicts = {}
    for key, value in dict_of_dfs.items():
        sum_dicts[key] = [sum(value.loc['Counts'])]
    df = pd.DataFrame(sum_dicts)
    df.index = ['Counts']
    SUM = sum(df.loc['Counts'])
    Probs = []
    for col in df.columns:
        c = float(df[col].tolist()[0])
        Probs.append(c/SUM)
    df.loc['Probabilities'] = Probs
    def extract_number(column_name):
        # Find all numbers at the end of the string
        match = re.search(r'(\d+)$', column_name)
        if match:
            return int(match.group(1))  # Return the number as an integer
        return float('inf')
    sorted_columns = sorted(df.columns, key=extract_number)
    df_sorted = df[sorted_columns]
    df_sorted.to_csv(os.path.join(outpath, 'epitope_binding_probablity.csv'), index=False)
    plt.figure(figsize=(10, 7))
    plt.plot(df_sorted.columns, df_sorted.loc['Probabilities'], marker='o', color='black')
    # Step 3: Customize plot
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.xlabel("Epitope Amino Acids", size=12)
    plt.ylabel(f"Probablity of Binding", size=12)
    plt.title(name, size=14)
    # Display the plot
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, 'Epitope_Binding_Probablity_to_Ab.png'), dpi=600)
    plt.close()

def paratope_prob_plot(DF, plotpath, type):
    plt.figure(figsize=(16, 8))
    plt.plot(DF.columns, DF.loc['Probs'], marker='o', color='black')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.xlabel("Paratope Residues", size=10)  # X-axis label
    plt.ylabel("Binding Probabilities", size=10)  # Y-axis label
    plt.title(f"Paratope {type} chain Binding Probabilities", size=14)
    plt.grid(True)  # Show grid for better readability
    plt.savefig(os.path.join(plotpath, f'paratope_probs_{type}.png'), dpi=600)
    plt.close()
    
def split_and_renumber_pdb(pdb_filename, output_filename, gap_threshold=200):
    """
    Splits a single-chain PDB into multiple chains based on residue number gaps
    and renumbers all residues sequentially from 1 to total length.

    Parameters:
        pdb_filename (str): Input PDB file path.
        output_filename (str): Output PDB file path.
        gap_threshold (int): Residue gap threshold to split chains.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_filename)

    # Create a new structure to hold the updated chains and residues
    new_structure = PDB.Structure.Structure("split_structure")
    new_model = PDB.Model.Model(0)  # Assume single model
    new_structure.add(new_model)

    current_chain = None
    last_residue_number = None
    residue_counter = 1
    # Loop through residues in the original structure
    for model in structure:
        for chain in model:
            for residue in chain:
                resnum = residue.id[1]

                # Check if a new chain should be started
                if last_residue_number is None or resnum - last_residue_number >= gap_threshold:
                    # Create a new chain
                    chain_id = chr(65 + len(new_model))  # Chain IDs: A, B, C, ...
                    current_chain = PDB.Chain.Chain(chain_id)
                    new_model.add(current_chain)

                # Create a new residue with renumbered ID
                new_residue = PDB.Residue.Residue(
                    (' ', residue_counter, ' '), residue.get_resname(), residue.get_segid()
                )
                # Add all atoms from the original residue
                for atom in residue:
                    new_residue.add(atom)

                # Add the new residue to the current chain
                current_chain.add(new_residue)

                # Update counters and tracking variables
                residue_counter += 1
                last_residue_number = resnum
        # Renumber residues within each chain starting from 1
    for chain in new_model:
        residue_counter = 1
        for residue in chain:
            residue.id = (' ', residue_counter, ' ')
            residue_counter += 1
    
                
    # Save the updated structure
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_filename)
    
    
def plot_bar(sub_df, output=None, show=False):
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(5, 5))
    x = []
    y = []
    for i, column in enumerate(sub_df.columns):
        x.append(i)
        y.append(np.mean(sub_df[column].values))
    ax.bar(x, y, alpha=1.0, color='lightgrey', width=0.4, edgecolor='black')
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('pLDDT', fontsize=12)
    ax.set_title('Average Alphafold Model Performace', fontsize=12)
    ax.set_xticks(range(len(sub_df.columns)))
    ax.set_xticklabels([i.replace('_1tmp_plddt', ' 1 template').replace('_plddt', ' 4 templates') for i in sub_df.columns], rotation=45, ha='right')
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=1200)
    if show:
        plt.show()
    plt.close()
        
def compute_aa_frequencies(msa_file):
    """Computes the frequency of each amino acid at each column in the MSA."""
    
    # Read MSA sequences
    sequences = [str(record.seq) for record in SeqIO.parse(msa_file, "fasta")]
    
    if not sequences:
        raise ValueError("MSA file is empty or improperly formatted.")
    
    # Define standard amino acids (excluding gaps)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    # Get alignment length (assuming all sequences are the same length)
    alignment_length = len(sequences[0])
    
    # Initialize frequency matrix
    freq_matrix = np.zeros((20, alignment_length))
    
    # Count amino acids per position
    for i in range(alignment_length):
        column_residues = [seq[i] for seq in sequences if seq[i] in amino_acids]  # Ignore gaps
        
        # Compute frequencies
        total_residues = len(column_residues)
        if total_residues > 0:
            for j, aa in enumerate(amino_acids):
                freq_matrix[j, i] = column_residues.count(aa) / total_residues  # Normalize
    
    # Convert to DataFrame for readability
    freq_df = pd.DataFrame(freq_matrix, index=list(amino_acids), columns=[f"Pos_{i+1}" for i in range(alignment_length)])
    
    return freq_df


def compute_conservation_percentage(msa_file):
    """Computes conservation percentage for each column in an MSA using Shannon entropy."""
    
    # Read MSA sequences
    sequences = [str(record.seq) for record in SeqIO.parse(msa_file, "fasta")]
    
    if not sequences:
        raise ValueError("MSA file is empty or improperly formatted.")
    
    # Define standard amino acids (excluding gaps)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    # Get alignment length
    alignment_length = len(sequences[0])
    
    # Initialize frequency matrix
    freq_matrix = np.zeros((20, alignment_length))
    
    # Compute frequencies
    for i in range(alignment_length):
        column_residues = [seq[i] for seq in sequences if seq[i] in amino_acids]  # Ignore gaps
        total_residues = len(column_residues)
        
        if total_residues > 0:
            for j, aa in enumerate(amino_acids):
                freq_matrix[j, i] = column_residues.count(aa) / total_residues  # Normalized frequency
    
    # Compute Shannon entropy H(i)
    entropy_values = -np.nansum(freq_matrix * np.log2(freq_matrix, where=freq_matrix > 0), axis=0)
    
    # Compute conservation score (1 - H(i)/H_max)
    H_max = np.log2(20)  # Maximum entropy (~4.32 for 20 amino acids)
    conservation_scores = 1 - (entropy_values / H_max)
    
    # Convert to DataFrame
    conservation_df = pd.DataFrame(conservation_scores, index=[f"Pos_{i+1}" for i in range(alignment_length)], columns=["Conservation Score"])
    
    return conservation_df.transpose()


# ═══════════════════════════════════════════════════════════════════
# BLOSUM62 Matrix (standard 23 amino acids including ambiguous codes)
# ═══════════════════════════════════════════════════════════════════
BLOSUM62 = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1],
    'B': [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1],
    'Z': [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'X': [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1],
}

# ═══════════════════════════════════════════════════════════════════
# Uniform background Frequency Background Frequencies
# These are the standard background frequencies derived from BLOSUM62
# Source: Henikoff & Henikoff (1992)
# ═══════════════════════════════════════════════════════════════════
BLOSUM62_BACKGROUND_FREQ = {
    'A': 0.05,
    'R': 0.05,
    'N': 0.05,
    'D': 0.05,
    'C': 0.05,
    'Q': 0.05,
    'E': 0.05,
    'G': 0.05,
    'H': 0.05,
    'I': 0.05,
    'L': 0.05,
    'K': 0.05,
    'M': 0.05,
    'F': 0.05,
    'P': 0.05,
    'S': 0.05,
    'T': 0.05,
    'W': 0.05,
    'Y': 0.05,
    'V': 0.05,
}

# Standard 20 amino acids
STANDARD_AA = list('ACDEFGHIKLMNPQRSTVWY')


def compute_position_specific_msa_score(msa_freq_df, pseudocount=0.01):
    """
    Compute position-specific MSA score P(k, j) for all positions and amino acids.
    
    P(k, j) = 2 * log2( (f_k(j) + pseudocount) / q_j )
    
    Where:
    - f_k(j) = observed frequency of amino acid j at position k in the MSA
    - q_j = background frequency of amino acid j (BLOSUM62 background)
    - pseudocount = small value to avoid log(0), default 0.01
    
    NOTE: For more sophisticated pseudocount strategies (PSSM-style), you could
    integrate data-dependent pseudocounts based on number of sequences in the MSA
    or use position-specific background frequencies. The current implementation
    uses a simple uniform pseudocount for computational efficiency and robustness.
    
    Alternative pseudocount strategies could be implemented here:
    1. Frequency-weighted: pseudocount_j = alpha * q_j where alpha ~ sqrt(N_sequences)
    2. Dirichlet mixture priors: Use mixture of Dirichlet distributions
    3. Henikoff-style: Weight pseudocounts by sequence diversity
    
    Args:
        msa_freq_df: DataFrame with amino acids as rows, positions as columns
                     containing observed frequencies f_k(j)
        pseudocount: Uniform pseudocount value (default 0.01)
    
    Returns:
        DataFrame with same shape containing P(k, j) scores
        Columns are positions (Pos_1, Pos_2, ...), rows are amino acids
    """
    # Initialize output DataFrame
    p_scores = pd.DataFrame(
        index=STANDARD_AA,
        columns=msa_freq_df.columns,
        dtype=float
    )
    
    for pos in msa_freq_df.columns:
        for aa in STANDARD_AA:
            # Get observed frequency at this position
            f_kj = msa_freq_df.loc[aa, pos] if aa in msa_freq_df.index else 0.0
            
            # Get background frequency
            q_j = BLOSUM62_BACKGROUND_FREQ[aa]
            
            # Compute P(k, j) = 2 * log2( (f_k(j) + pseudocount) / q_j )
            # Note: Higher P(k,j) means amino acid j is MORE common at position k
            # relative to background, so mutations TO j would be penalized
            p_kj = 2.0 * np.log2((f_kj + pseudocount) / q_j)
            
            p_scores.loc[aa, pos] = p_kj
    
    return p_scores


def compute_psbdm_matrix(sequence, msa_freq_df, weight=1.0, pseudocount=0.01):
    """
    Compute Position-Specific BLOSUM Dissimilarity Matrix (PSBDM).
    
    For each position k with wild-type amino acid i, compute the combined score
    for all 19 possible mutant amino acids j:
    
    Score(j) = B(i, j) + [w * P(k, j)]
    
    Where:
    - B(i, j) = BLOSUM62 score between wild-type i and mutant j
    - w = weight parameter (default 1.0)
    - P(k, j) = position-specific MSA score = 2 * log2( (f_k(j) + pseudocount) / q_j )
    
    LOWER scores indicate BETTER mutations (more dissimilar + rare at that position).
    
    Args:
        sequence: Antibody sequence string
        msa_freq_df: DataFrame with amino acid frequencies from MSA
        weight: Weight parameter for MSA term (default 1.0)
        pseudocount: Pseudocount for log calculation (default 0.01)
    
    Returns:
        psbdm_df: DataFrame with amino acids as rows, positions as columns
                  Values are combined PSBDM scores
        p_scores_df: DataFrame with raw P(k, j) values
    """
    # First compute P(k, j) for all positions
    p_scores = compute_position_specific_msa_score(msa_freq_df, pseudocount=pseudocount)
    
    # Get BLOSUM keys for indexing
    blosum_keys = list(BLOSUM62.keys())
    
    # Initialize output DataFrame
    psbdm = {}
    
    for pos, aa_wt in enumerate(sequence):
        col = f'Pos_{pos + 1}'
        
        if col not in p_scores.columns:
            # If position not in MSA, skip (shouldn't happen normally)
            continue
        
        scores = []
        for aa_mut in STANDARD_AA:
            if aa_mut == aa_wt:
                # Wild-type to wild-type: mark as NaN
                scores.append(np.nan)
            else:
                # Get BLOSUM62 score B(wt, mut)
                wt_idx = blosum_keys.index(aa_wt)
                mut_idx = blosum_keys.index(aa_mut)
                b_score = BLOSUM62[aa_wt][mut_idx]
                
                # Get position-specific MSA score P(k, mut)
                p_score = p_scores.loc[aa_mut, col]
                
                # Combined score: Score(mut) = B(wt, mut) + w * P(k, mut)
                # Lower is better (more dissimilar + rarer in MSA)
                combined = b_score + (weight * p_score)
                scores.append(combined)
        
        psbdm[col] = scores
    
    psbdm_df = pd.DataFrame(psbdm, index=STANDARD_AA)
    
    return psbdm_df, p_scores


def get_blosum_score(aa1, aa2):
    """Get BLOSUM62 score between two amino acids."""
    blosum_keys = list(BLOSUM62.keys())
    return BLOSUM62[aa1][blosum_keys.index(aa2)]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def blosum_weighted_score(aa_1, aa_2, freq1, freq2):
    if aa_1 == aa_2: 
        return 0.
    else:
        bl1_ind = list(BLOSUM62.keys()).index(aa_1)
        bl2_ind = list(BLOSUM62.keys()).index(aa_2)
        bl1 = BLOSUM62[aa_1][bl1_ind]
        bl2 = BLOSUM62[aa_1][bl2_ind]
        score = 1 - np.exp(0.3*(bl2 - bl1))
        weight = np.exp(-freq2/(freq1 + 1e-9))
        return score * weight

def calculate_blossum_weighted_score(ab_sequence, frequency):
    amino_acids = list(frequency.index)
    DICT = {}
    identicals = []
    for pos, aa_1 in enumerate(ab_sequence):
        ind = amino_acids.index(aa_1)
        freq1 = frequency[f'Pos_{pos+1}'].tolist()[ind]
        identicals.append(np.exp(-15*freq1))
        col_score = []
        for aa_2 in amino_acids:
            freq2 = frequency.loc[aa_2, f'Pos_{pos+1}']
            bls = blosum_weighted_score(aa_1, aa_2, freq1, freq2)
            col_score.append(bls)
        DICT[f'Pos_{pos+1}'] = col_score
    DICT = pd.DataFrame(DICT)
    DICT.index = amino_acids
    return DICT, identicals

def get_residue_coordinates(residue, atom_types=['CA', 'CB', 'N', 'O', 'C']):
    """
    Attempt to get coordinates of a residue, trying different atom types in order.
    """
    for atom_type in atom_types:
        if atom_type in residue:
            return residue[atom_type].coord
    return []


def calculate_distance_score_from_pdb(pdb_filename, subset_chain, threshold=15., step_thr=None):
    if not step_thr: step_thr = [4., 5., 6., 7., 8.]
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_filename)

    # retreive coordinates
    ref_chain = list(subset_chain.keys())[0]
    ref_res = list(subset_chain.values())[0]

    coords_dict = {}
    ref_dict = {}
    for model in structure:
        for chain in model:
            coords = []
            for residue in chain:
                coord = get_residue_coordinates(residue)
                if len(coord) != 3: 
                    raise ValueError(f'coordinate for {residue} not found, returned coords: {coord}')
                if chain.id == ref_chain and residue.id[1] in ref_res:
                    ref_dict[f'{residue.id[1]}'] = [residue, coord]
                coords.append(coord)
            coords_dict[chain.id] = coords
    ref_list = [value[1] for key, value in ref_dict.items()]
    Distance_Scores = {}
    Score_matrix = {}
    for chain, values in coords_dict.items():
        if chain == ref_chain: 
            continue
        dist = scipy.spatial.distance.cdist(np.array(values), np.array(ref_list))
        C = []
        # calculate distance score via step function
        for step in step_thr + [threshold]:
            contact = np.expand_dims(np.where(dist <= step, 1., 0.), axis=-1)
            C.append(contact)
        C = np.sum(np.concatenate(C, axis=-1), axis=-1) / (len(step_thr) + 1)
        sigmoid_vectorized = np.vectorize(sigmoid)
        Score_matrix[chain] = C
        C = sigmoid_vectorized(np.sum(C, axis=-1)) - 0.5
        C = np.clip(C, 0., 1.)
        Distance_Scores[chain] = C

    return Distance_Scores, Score_matrix # dict[H,L] (N,), dict[H,L] no sigmoid (N,E)

def calculate_distance_score_from_folder(pdb_folder_path):
    files = [os.path.join(pdb_folder_path, i) for i in os.listdir(pdb_folder_path) if '.pdb' in i]
    Distance_scores_heavy = []
    Distance_scores_light = []
    Score_matrix_heavy = []
    Score_matrix_light = []
    for pdb_file_path in files:
        distance_score, score_matrix = calculate_distance_score_from_pdb(pdb_file_path, 
                                                                   subset_chain = {'G':list(range(29,46))}, 
                                                                   threshold=12., 
                                                                   step_thr=None)
        Distance_scores_heavy.append(distance_score['H'][:,np.newaxis])
        Distance_scores_light.append(distance_score['L'][:,np.newaxis])
        Score_matrix_heavy.append(score_matrix['H'][:, :, np.newaxis])
        Score_matrix_light.append(score_matrix['L'][:, :, np.newaxis])
        
    heavy_chain_dist_score = np.mean(np.concatenate(Distance_scores_heavy, axis=-1), axis=-1)
    light_chain_dist_score = np.mean(np.concatenate(Distance_scores_light, axis=-1), axis=-1)
    score_matrix_heavy_score = np.mean(np.concatenate(Score_matrix_heavy, axis=-1), axis=-1)
    score_matrix_light_score = np.mean(np.concatenate(Score_matrix_light, axis=-1), axis=-1)
    return ({'H':heavy_chain_dist_score, 'L':light_chain_dist_score}, 
            {'H':score_matrix_heavy_score, 'L':score_matrix_light_score})

def calculate_paratope_probability_score(df):
    no_interaction_prob = 1 - df.loc[['pt', 'hy', 'hb'], :].to_numpy()
    no_interaction_prob = np.prod(no_interaction_prob, axis=0)
    no_interaction_prob = 1 - no_interaction_prob
    return no_interaction_prob

def calculate_conservation_score(cons_df) :
    sigmoid_vectorized = np.vectorize(sigmoid)
    cons = cons_df.to_numpy().flatten()
    score = 2.5 * (-np.log(cons + 1e-9) - 1.)
    score = sigmoid_vectorized(score)
    return score
