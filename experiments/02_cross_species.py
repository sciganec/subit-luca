# experiments/02_cross_species.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from encoder.universal import UniversalEncoder
from datasets.human import load_human_pbmc
from datasets.bacteria import load_ecoli_synthetic
from metrics.entropy import entropy
from metrics.luca import luca_distance

def main():
    os.makedirs("results/cross_species", exist_ok=True)
    
    # Load data
    adata_human = load_human_pbmc(normalize=False)
    adata_human.obs['species'] = 'Human'
    adata_bac = load_ecoli_synthetic(n_cells=200)
    adata_bac.obs['species'] = 'E. coli'
    
    # Encode human (exact gene symbols, normalise)
    enc_human = UniversalEncoder(use_substring=False, normalize=True)
    P64_human = enc_human.encode(adata_human)
    
    # Encode bacteria (use substring, data already log1p -> no extra normalisation)
    enc_bac = UniversalEncoder(use_substring=True, normalize=False)
    P64_bac = enc_bac.encode(adata_bac)
    
    # Combine the 64‑D probability matrices
    X_combined = np.vstack([P64_human, P64_bac])
    adata_comb = sc.AnnData(X_combined)
    adata_comb.obs['species'] = ['Human']*adata_human.n_obs + ['E. coli']*adata_bac.n_obs
    
    # Compute metrics on the combined SUBIT space
    adata_comb.obs['entropy'] = entropy(X_combined)
    adata_comb.obs['luca_dist'] = luca_distance(X_combined)
    adata_comb.obs['complexity'] = adata_comb.obs['entropy'] + adata_comb.obs['luca_dist']
    
    # UMAP on the 64‑D probabilities
    sc.pp.neighbors(adata_comb, n_pcs=None, use_rep='X')
    sc.tl.umap(adata_comb)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.umap(adata_comb, color='species', ax=axes[0], show=False, title='Map of Life')
    sc.pl.umap(adata_comb, color='complexity', ax=axes[1], show=False, title='Complexity')
    plt.tight_layout()
    plt.savefig('results/cross_species/umap_cross_species.png', dpi=150)
    plt.show()
    
    # Save combined object
    adata_comb.write('results/cross_species/combined.h5ad')
    print("Experiment completed. Results saved in results/cross_species/")

if __name__ == '__main__':
    main()