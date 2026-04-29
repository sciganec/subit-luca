# SUBIT-LUCA: 64 States of Life – A Universal Coordinate System for Any Cell

> **Every cell has a 6‑bit address. LUCA is `000000`. Evolution is the sequential turning on of bits.**

**SUBIT-LUCA** is a theoretical and computational framework that describes the state of any living cell (eukaryote, bacterium, archaeon) using three biological dimensions – **WHO**, **WHERE**, **WHEN** – each encoded by 2 bits. Together they form **64 archetypes** (states), and their dynamics define the cell cycle, homeostasis, and evolution.

The project provides:
- 🧬 **Code** to transform scRNA‑seq data into 64 states (P(SUBIT))
- 🌍 **Universal encoder** that works across species (human, bacteria) via functional patterns
- 🧭 **Evolutionary levels** from LUCA (18 states) to full eukaryote (64 states)
- 📊 **Visualisations**: 3D cube, heatmaps, UMAP “map of life”
- 🔄 **Simulation** of Markovian transitions between states

---

## 🧠 The Map of Life (SUBIT Map)

Three dimensions, each with four values:

| Dimension | Bits | Values | Physical meaning |
|-----------|------|--------|------------------|
| **WHO** (subject) | b1b2 | `10` = ME <br> `11` = WE <br> `01` = YOU <br> `00` = THEY | single organelle<br>network/collective<br>neighbour signal<br>background/matrix |
| **WHERE** (space) | b3b4 | `10` = EAST <br> `11` = SOUTH <br> `01` = WEST <br> `00` = NORTH | cytosol/nucleus<br>membrane<br>extracellular<br>boundary/pore |
| **WHEN** (time) | b5b6 | `10` = SPRING <br> `11` = SUMMER <br> `01` = AUTUMN <br> `00` = WINTER | G1 growth<br>S replication<br>G2/mitosis<br>G0/apoptosis |

**Example**: state `11 11 11` (WE + SOUTH + SUMMER) — replisome on the nuclear membrane during S‑phase.  
Code `10 10 10` (ME + EAST + SPRING) — a mitochondrion in the cytosol during growth phase.

Total: **2⁶ = 64 archetypes**.

---

## 🧬 Evolutionary levels (as code)

Evolution is implemented as a **monotonically increasing set of allowed states**:

| Level | Name | Added compared to previous | Number of states |
|-------|------|----------------------------|------------------|
| 0 | LUCA | THEY, WE; EAST, SOUTH; SPRING, SUMMER, AUTUMN | 18 |
| 1 | Prokaryote | + YOU (signalling) | 24 |
| 2 | Endosymbiosis | + ME (organelles, mitochondria) | 32 |
| 3 | Nucleus | + WEST, NORTH (compartments, pores) | 48 |
| 4 | Control | + WINTER (apoptosis, G0) | 56 |
| 5 | Eukaryote | all states | 64 |

Each level is a separate class in `levels/` implementing `allowed_states()`. Monotonicity is enforced by tests.

---

## ⚙️ Installation

```bash
git clone https://github.com/sciganec/subit-luca.git
cd subit-luca
pip install -e .
```

Or via pip (after publication):
```bash
pip install subit-luca
```

Minimal dependencies: `numpy`, `scipy`, `scanpy` (optional, for data handling).

---

## 🚀 Quick start (30 seconds)

```python
import scanpy as sc
from subit.encoder import UniversalEncoder
from subit.metrics import complexity, luca_distance

# Load test data (PBMC 3k)
adata = sc.datasets.pbmc3k()

# Build P(SUBIT) – 64 states for each cell
encoder = UniversalEncoder()
P64 = encoder.encode(adata)            # (n_cells, 64)
adata.obsm["X_subit64"] = P64

# Compute evolutionary complexity
adata.obs["complexity"] = complexity(P64)
adata.obs["luca_dist"] = luca_distance(P64)

# UMAP in SUBIT space
sc.pp.neighbors(adata, use_rep="X_subit64")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["complexity", "luca_dist", "phase"])
```

Result: a UMAP where different cell types obtain coordinates that reflect their internal state according to SUBIT.

---

## 📊 Example visualisation: 3D cube for a single cell

```python
from subit.viz import plot_subit_cube

# Take the first cell
prob = adata.obsm["X_subit64"][0]
plot_subit_cube(prob, title="PBMC cell: SUBIT distribution")
```

(Opens an interactive Plotly graph where each of the 64 archetypes is a coloured point at WHO, WHERE, WHEN coordinates.)

---

## 🧭 Universal map of life (human + bacteria)

`UniversalEncoder` uses functional patterns (translation, energy, membrane, stress) instead of specific genes. This allows comparing species with different genomes.

```python
from subit.datasets import load_human_pbmc, load_ecoli
from subit.encoder import UniversalEncoder

adata_human = load_human_pbmc()
adata_ecoli = load_ecoli()        # synthetic or real data

# Concatenate and encode
import scanpy as sc
adata_all = adata_human.concatenate(adata_ecoli)
encoder = UniversalEncoder()
P64_all = encoder.encode(adata_all)
adata_all.obsm["X_subit64"] = P64_all

# UMAP coloured by species
sc.pp.neighbors(adata_all, use_rep="X_subit64")
sc.tl.umap(adata_all)
sc.pl.umap(adata_all, color=["species"])
```

Expected result: a **continuum** from bacteria (low complexity, close to LUCA) to eukaryotes (high complexity).

---

## 📚 Documentation

- [SUBIT theory](docs/theory.md) – mathematics of 64 states and locality principle
- [Universal mapping](docs/universal_mapping.md) – how functional patterns replace genes
- [LUCA reconstruction](docs/luca.md) – justification of the 18 states
- [Evolutionary levels](docs/evolution_levels.md) – design of `levels/`
- [API reference](docs/api.md) – auto-generated documentation

---

## 🔬 Reproducible experiments

All main results can be obtained by running scripts in `experiments/`:

```bash
python experiments/01_luca_projection.py     # project modern cells onto LUCA space
python experiments/02_subit_umap.py          # UMAP of 64 states
python experiments/03_cross_species.py       # human+bacteria map of life
python experiments/04_phylogeny.py           # phylogeny tree based on SUBIT distances
```

Each script saves figures and tables into `results/`.

---

## 🛠️ Development and testing

```bash
pip install -e .[dev]
pytest tests/
```

Before committing, ensure tests pass and level monotonicity holds.

---

License: MIT – free for academic and commercial use.

---

*“The cell is no longer a black box – it becomes a finite automaton that remembers its LUCA ancestor in every bit.”*
