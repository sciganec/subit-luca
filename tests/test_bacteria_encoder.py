from datasets.bacteria import load_ecoli_synthetic
from encoder.universal import UniversalEncoder
import numpy as np

adata = load_ecoli_synthetic(n_cells=200)
print(f"E. coli synthetic: {adata.n_obs} cells, {adata.n_vars} genes")

enc = UniversalEncoder(use_substring=True, normalize=False)  # дані вже log1p
P64 = enc.encode(adata)
print(f"P64 shape: {P64.shape}")
print(f"Row sums: {P64.sum(axis=1)[:5]}")
assert np.allclose(P64.sum(axis=1), 1.0, atol=1e-5)
print("Bacteria encoder test PASSED")