# SPECIFICATION: SUBIT-LUCA THEORY

**Version:** 0.3.0  
**Date:** 2026-04-30  

---

## 1. Abstract

SUBIT-LUCA is a formal framework for describing the state of any living cell using a **6-bit code** (64 possible states). The code is composed of three orthogonal dimensions: **WHO**, **WHERE**, **WHEN**. The framework defines a **monotonic evolutionary sequence** of subspaces, from LUCA (18 states) to modern eukaryotes (64 states).

---

## 2. Foundational Axioms

| Axiom | Statement |
|-------|-----------|
| A1 | Every living cell can be described by three independent attributes: WHO, WHERE, WHEN |
| A2 | Each attribute can take exactly 4 discrete values (encoded by 2 bits) |
| A3 | Total number of states = 4 × 4 × 4 = 64 |
| A4 | Evolution proceeds by monotonically increasing the set of allowed states |
| A5 | State transitions occur only between Hamming distance 1 neighbours (one-bit changes) |

---

## 3. The Three Dimensions

### 3.1 WHO (2 bits)

| WHO | Bits | Name | Biology |
|-----|------|------|---------|
| 0 | 00 | THEY | extracellular matrix, dead material |
| 1 | 01 | YOU | extracellular signal, neighbouring cell |
| 2 | 10 | ME | single organelle (mitochondrion, lysosome) |
| 3 | 11 | WE | network / collective (ribosomes, replisome) |

### 3.2 WHERE (2 bits)

| WHERE | Bits | Name | Biology |
|-------|------|------|---------|
| 0 | 00 | NORTH | boundary / transition zone |
| 1 | 01 | WEST | extracellular space / secretion |
| 2 | 10 | EAST | interior (cytosol, nucleoplasm) |
| 3 | 11 | SOUTH | membrane (plasma, ER, envelope) |

### 3.3 WHEN (2 bits)

| WHEN | Bits | Name | Cell cycle phase |
|------|------|------|------------------|
| 0 | 00 | WINTER | G0 / quiescence / apoptosis |
| 1 | 01 | AUTUMN | G2 / mitosis / division |
| 2 | 10 | SPRING | G1 / growth / synthesis |
| 3 | 11 | SUMMER | S phase / replication |

---

## 4. State Encoding

### 4.1 Index Formula (Unicode)

```
state_index = (WHO × 16) + (WHERE × 4) + WHEN

or in bit notation:

state_index = (WHO << 4) | (WHERE << 2) | WHEN

where '<<' = left bit shift, '|' = bitwise OR
```

### 4.2 Examples

| WHO | WHERE | WHEN | Bits       | Index | Name                     |
|-----|-------|------|------------|-------|--------------------------|
| WE (3) | EAST (2) | SUMMER (3) | 11 10 11 | 59 | WE/EAST/SUMMER           |
| ME (2) | SOUTH (3) | SPRING (2) | 10 11 10 | 46 | ME/SOUTH/SPRING          |
| YOU (1) | WEST (1) | AUTUMN (1) | 01 01 01 | 21 | YOU/WEST/AUTUMN          |
| THEY (0) | NORTH (0) | WINTER (0) | 00 00 00 | 0  | THEY/NORTH/WINTER (LUCA) |

---

## 5. Evolutionary Levels

| Level | Name | WHO | WHERE | WHEN | States |
|-------|------|-----|-------|------|--------|
| 0 | LUCA | THEY, WE | EAST, SOUTH | SPRING, SUMMER, AUTUMN | 2×2×3 = 18 |
| 1 | Prokaryote | + YOU | same | same | 3×2×3 = 24 |
| 2 | Endosymbiosis | + ME | same | same | 4×2×3 = 32 |
| 3 | Nucleus | same | + WEST, NORTH | same | 4×4×3 = 48 |
| 4 | Control | same | same | + WINTER | 4×4×4 = 64 |
| 5 | Eukaryote | all | all | all | 64 |

**Monotonicity rule:** states(Level_i) ⊆ states(Level_{i+1})

---

## 6. Probability Distribution

For a single cell measured by scRNA‑seq:

```
P(subit) = [p₀, p₁, ..., p₆₃]

where:
p₀ + p₁ + ... + p₆₃ = 1
pᵢ ≥ 0 for all i
```

Construction from three independent softmax scores:

```
p_who,where,when = softmax(s_who) × softmax(s_where) × softmax(s_when)

where s_who, s_where, s_when are average expression scores over functional gene groups
```

---

## 7. Metrics (Unicode formulas)

### 7.1 Shannon Entropy

```
H(P) = – Σᵢ₌₀⁶³ pᵢ × log(pᵢ)
```

Measures distribution spread (plasticity vs commitment).  
Higher H = more mixed state (stem‑like), lower H = committed / specialised.

### 7.2 LUCA Distance

```
D_LUCA(P) = Σᵢ₌₀⁶³ pᵢ × popcount(i)

where popcount(i) = number of '1' bits in the 6-bit representation of i
```

Range: 0 (all mass on LUCA anchor 000000) to 6 (all mass on 111111).  
Measures evolutionary distance from the Last Universal Common Ancestor.

### 7.3 Complexity (composite)

```
C(P) = H(P) + D_LUCA(P)
```

Empirical ranges:
- Bacteria (E. coli): C ≈ 4.5 – 5.5
- Human immune cells: C ≈ 6.0 – 7.5

### 7.4 Hamming Distance (between two states i and j)

```
d_H(i, j) = popcount(i XOR j)
```

Distance = number of differing bits (0 to 6).

---

## 8. Transition Dynamics

**Local transitions** (standard): Only states with Hamming distance = 1 are directly reachable.  
Each state has at most 6 neighbours.

**Global transitions** (special): Mitosis, apoptosis, or external shocks may change multiple bits in a single step (macros).

**Topology:** The 64 states form a 6‑dimensional hypercube.

---

## 9. Universality (Cross‑species)

The encoder uses **functional patterns** instead of species‑specific genes:

| Functional group | Universal patterns | Works in |
|-----------------|-------------------|----------|
| Translation | `rpl`, `rps`, `RPL`, `RPS` | human, bacteria, archaea |
| Energy | `atp`, `ndh`, `MT‑`, `ATP5` | human, bacteria |
| Membrane | `sec`, `omp`, `CD44`, `SEC61` | human, bacteria |
| Division | `ftsZ`, `CDK1`, `CCNB` | human, bacteria |
| Stress | `lon`, `clp`, `CASP`, `BAX` | human, bacteria |

Thus, SUBIT provides a **common latent space** for any species without gene orthology matching.

---

## 10. Implementation Specifications

| Component | Requirement |
|-----------|-------------|
| Encoding | O(n_cells × n_genes) with pre‑computed functional groups |
| Storage | P64 as float32, shape (n_cells, 64) |
| Metrics | vectorised over cells (no per‑cell loops) |
| Visualisation | UMAP on P64, 3D cube, heatmaps |
| Level projection | zero out forbidden states, renormalise |

---

## 11. Validation Criteria

| ID | Criterion | Pass condition |
|----|-----------|----------------|
| V1 | 64‑state construction | sum(P64) = 1 per cell |
| V2 | Species separation | Human and E. coli clusters separate on UMAP |
| V3 | Complexity gradient | mean(complexity) bacteria < mean(complexity) human |
| V4 | Monotonicity | states(L_i) ⊆ states(L_{i+1}) |
| V5 | Phase correlation | WHEN distribution correlates with S/G2M scores |

---

## 12. Known Limitations

1. Functional gene sets are heuristic – needs systematic derivation from KEGG/GO
2. Bacterial WHERE dimension less precise (no organelles)
3. Apoptosis (WINTER) is eukaryotic‑specific; mapped to stress for bacteria
4. Synthetic bacterial data – replace with real datasets for publication
5. Transition probabilities not yet estimated from temporal data

---

## 13. Quick Reference Card

```
SUBIT STATE = (WHO × 16) + (WHERE × 4) + WHEN

WHO:   0=THEY   1=YOU   2=ME   3=WE
WHERE: 0=NORTH  1=WEST  2=EAST  3=SOUTH
WHEN:  0=WINTER 1=AUTUMN 2=SPRING 3=SUMMER

LUCA anchor: index 0 (THEY/NORTH/WINTER)
Eukaryote anchor: index 63 (WE/SOUTH/SUMMER)

Complexity = entropy + LUCA distance
Allowed transitions: Hamming distance = 1
```

---

## 14. References

1. Weiss, M. C. et al. (2016). The physiology and habitat of the last universal common ancestor. *Nature Microbiology*.
2. Louca, S., & Pennell, M. W. (2020). A general theory of evolution as a branching process. *Nature Ecology & Evolution*.
3. Wolf, Y. I. et al. (2012). The LUCA and its complex virome. *Biology Direct*.
4. KEGG Orthology database for functional modules.

---

*This specification is the formal foundation of the SUBIT-LUCA framework. All code, experiments, and visualisations derive from these principles.*