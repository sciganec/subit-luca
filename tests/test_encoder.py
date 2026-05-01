import scanpy as sc
from encoder.universal import UniversalEncoder
from metrics.entropy import entropy
from metrics.luca import luca_distance

# Завантажити маленький датасет
adata = sc.datasets.pbmc3k()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Енкодер (сам зробить нормалізацію та log1p)
enc = UniversalEncoder(normalize=True)
P64 = enc.encode(adata)

print("Shape P64:", P64.shape)         # має бути (n_cells, 64)
print("Sum of probabilities per cell:", P64.sum(axis=1)[:5])  # має бути 1

# Метрики
print("Entropy (nats):", entropy(P64)[:3])
print("LUCA distance:", luca_distance(P64)[:3])