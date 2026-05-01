import numpy as np
import scanpy as sc
from encoder.universal import UniversalEncoder
from datasets.human import load_human_pbmc

adata = load_human_pbmc(normalize=False)
print(f"Human data: {adata.n_obs} cells, {adata.n_vars} genes")

enc = UniversalEncoder(use_substring=False, normalize=True)
P64 = enc.encode(adata)
print(f"P64 shape: {P64.shape}")
print(f"Row sums: {P64.sum(axis=1)[:5]}")
assert np.allclose(P64.sum(axis=1), 1.0, atol=1e-5)
print("Human encoder test PASSED")