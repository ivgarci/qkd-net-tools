# qkd-net-tools — QKD Network Analysis Toolkit

**Doctoral dissertation (ES):** *Generación y validación de redes QKD a gran escala bajo restricciones físicas y tecnológicas*  
**English title:** *Generating and validating large-scale QKD networks under realistic physical and technological constraints*

**Author:** Iván García Cobo · Universidad de Salamanca · 2019–2026  
**Doctoral publication:** García-Cobo, I. & Menéndez, H.D. (2021). *Designing large quantum key distribution networks via medoid-based algorithms.* Future Generation Computer Systems, 115, 814–824.

---

This repository contains Python and R scripts for three thesis case studies:
generation of trusted-relay QKD topology candidates from territorial data
(Castile and León and Peninsular Spain), analysis of a topological proxy
derived from ADIF open railway data, and structural validation via complex
network analysis. The ADIF data used here do not establish the existence,
continuity or availability of dark fibre.

---

## Repository structure | Estructura del repositorio

```
qkd-net-tools/
├── analisis/          Network metrics: centrality, clustering, connectivity, degree distribution
├── ataques/           Resilience simulations: random failures and targeted attacks
├── adif/              ADIF-derived railway proxy: junction graph, resilience, figures, maps
├── generacion/        Topology generation: k-medoids (R/PAM), network builder (Python)
├── datos/             Input data files
│   ├── cyl/           Castile and León: adjacency matrix, node measures, failure/attack results
│   │   ├── cyl_1000.csv                          — 267 candidate municipalities
│   │   ├── AdjacencyMatrixNamed45.csv             — 100×100 adjacency (Δ=45 km)
│   │   ├── AdjacencyMatrixNamed45_exp.csv         — alternative parameter experiment
│   │   ├── Node_Specific_Network_Measures.csv     — per-node centrality, clustering, community
│   │   ├── Node_Specific_Network_Measures_exp.csv — same, alternative experiment
│   │   ├── random_failure_results.csv             — S(p) over R=300 random failure trials
│   │   ├── random_failure_results_exp.csv         — same, alternative experiment
│   │   ├── incremental_targeted_attack_results.csv     — targeted attack curve (0–49%, step 1%)
│   │   ├── incremental_targeted_attack_results_exp.csv — same, alternative experiment
│   │   └── targeted_attack_results.csv            — targeted attack (alternative format)
│   ├── espana/        Peninsular Spain: adjacency matrix, node measures, failure/attack results
│   │   ├── peninsula_1000.csv                     — 3,102 candidate municipalities
│   │   ├── AdjacencyMatrixNamed45.csv             — 950×950 adjacency (Δ=45 km)
│   │   ├── componentes_parametros.csv             — connected components by parameter sweep
│   │   ├── Node_Specific_Network_Measures.csv     — per-node centrality, clustering, community
│   │   ├── random_failure_results.csv             — S(p) over R=3,000 random failure trials
│   │   └── incremental_targeted_attack_results.csv — targeted attack curve (0–49%, step 1%)
│   ├── adif/          ADIF railway network data and analysis results
│   │   ├── nodos_red_adif.csv                     — 3,085 ADIF dependencies with coordinates
│   │   ├── adyacencia_red_adif.csv                — railway adjacency records with reported lengths
│   │   └── resultados_adif_junctions.json         — pre-computed metrics, failure/attack curves
│   ├── cyl_1000.csv          — (legacy path) Castile and León municipalities
│   └── peninsula_1000.csv    — (legacy path) Peninsular Spain municipalities
├── figuras/           Generated figures for all three case studies
│   ├── cyl/           CyL: degree distribution, Girvan-Newman communities, HJ-biplot, medoid map, network topology
│   ├── espana/        España: degree distribution, Girvan-Newman communities, HJ-biplot, medoid map, network topology, connected-components sweep
│   └── adif/          ADIF: georeferenced junction graph map, resilience curves S(p)
├── requirements.txt
└── LICENSE
```

---

## Requirements | Requisitos

**Python 3.10+** and **R 4.x** (for generation scripts only).

```bash
pip install -r requirements.txt
```

For the exact environment used to regenerate and audit the thesis SKR and CyL
routing artefacts:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-reproducibility.txt
.venv/bin/python -m pytest -q
.venv/bin/python protocols/skr_bb84.py
.venv/bin/python analisis/enrutamiento_qkd.py
```

| Package | Role |
|---------|------|
| `networkx` | Graphs and algorithms |
| `pandas` | CSV / adjacency matrices |
| `matplotlib` | Figures |
| `numpy` | Numerical operations |
| `scikit-learn` | PCA, KMeans in `analisis/` |
| `scipy` | Statistical tests |
| `python-louvain` | Louvain community detection |
| `geopandas` | Geospatial operations (ADIF map) |
| `folium` | Interactive HTML maps (optional) |

---

## Case studies and scripts | Casos de estudio y scripts

### Case I & II — CyL and Peninsular Spain (generated topologies)

| Script | Directory | Description |
|--------|-----------|-------------|
| `medoids.R` | `generacion/` | PAM k-medoids clustering; produces medoid assignments and adjacency matrix |
| `components.R` | `generacion/` | Connectivity verification of backbone graph |
| `plots.R` | `generacion/` | Topology visualisation |
| `generar_red.py` | `generacion/` | Python pipeline: Haversine distances, Δ-threshold edge formation, backbone construction |
| `analis_redes_complejas.py` | `analisis/` | Full structural analysis: centrality, PCA/KMeans, community detection, global metrics |
| `generar_grado_distribucion.py` | `analisis/` | Degree distribution histogram |
| `girvan_newmancyl.py` | `analisis/` | Girvan–Newman community detection with geo-layout |
| `centralidad.py` | `analisis/` | Degree / betweenness / closeness centrality charts |
| `coeficiente.py` | `analisis/` | Clustering coefficient distribution |
| `comunidad.py` | `analisis/` | Louvain community detection |
| `conectividad.py` | `analisis/` | Connectivity visualisation |
| `ataques_aleatorios_nodos_fault.py` | `ataques/` | Monte Carlo random-failure simulation (default: 13% of nodes, R=300, seed=42) |
| `ataques_dirigidos_nodos_fault.py` | `ataques/` | Incremental targeted attack by centrality (degree/betweenness/closeness), 0–49% in 1% steps |
| `metricas_avanzadas.py` | `analisis/` | Advanced global metrics: efficiency, assortativity, small-world σ, algebraic connectivity λ₂, scale-free exponent α |
| `comunidad.py` | `analisis/` | Louvain community detection (built-in NetworkX ≥3.0); saves `comunidades_louvain.pdf/.png` |
| `comparacion_resiliencia.py` | `ataques/` | Three-case S(p) comparison figure + unified robustness table (R, p*) + bootstrap 95% CI for random failures |
| `ataques_dinamicos.py` | `ataques/` | **Dynamic** targeted attack: betweenness recomputed after each removal; compares static vs dynamic p* |
| `ataques_aristas.py` | `ataques/` | Edge betweenness attack simulation; top-10 critical links; bridge identification |
| `k_core_decomposition.py` | `analisis/` | k-core hierarchy for all three cases; k-shell visualisation; `datos/k_core_decomposition.csv` |

**Input files** (place in `datos/` or update paths in scripts):

| File | Used by |
|------|---------|
| `datos/cyl_1000.csv` | `girvan_newmancyl.py`, `generar_red.py` |
| `datos/peninsula_1000.csv` | `generar_red.py` (Peninsular Spain case) |
| `AdjacencyMatrixNamed45.csv` | `analis_redes_complejas.py`, `ataques/`, `analisis/` |

### QKD physics and benchmarks

| Script | Directory | Description |
|--------|-----------|-------------|
| `skr_bb84.py` | `protocols/` | Ideal asymptotic BB84 SKR(d) model (Lo-Ma-Chen 2005; exact single-photon estimate, not a finite-decoy implementation); QBER(d); SKR per link for CyL/España; `figuras/skr_vs_distancia.pdf` |
| `enrutamiento_qkd.py` | `analisis/` | Deterministic key-aware routing over all 4,950 unordered CyL pairs: minimum hops (tie: maximum bottleneck) vs max-min SKR (tie: minimum hops); full paths in `datos/enrutamiento_qkd_allpairs.csv`, summary and top-10 bottlenecks; `figuras/comparacion_rutas_qkd.pdf` |
| `benchmarks_qkd.py` | `analisis/` | Compare same metrics with real published QKD networks (Tokyo 2011, SECOQC 2009, China 2021); `datos/benchmarks_qkd_comparacion.csv` |

---

### QKD-PAM-Generation (IEEE TNSE submission)

| Script | Description | Run |
|--------|-------------|-----|
| `analisis/tablas_paper_pam_generation.py` | Generates exact values for Tables II–IV of the PAM paper: k-means vs PAM coverage violations (Table II), Δ-sensitivity k* for CyL and España (Table III), network properties for k=80..120 in CyL (Table IV); requires `pip install scikit-learn-extra haversine` | `python analisis/tablas_paper_pam_generation.py` |
| `analisis/greedy_cds_cyl.py` | Greedy MCDS baseline on CyL (n=267, Δ=45 km): greedy MDS via set-cover then Steiner-path repair; yields k_greedy=42 | `python analisis/greedy_cds_cyl.py` |
| `analisis/ilp_mcds_cyl.py` | ILP lower bounds for MCDS on CyL: LP relaxation (k_LP=22.1), ILP minimum dominating set (k_DS=20), ILP MCDS (k_MCDS=46); requires `pip install pulp` | `python analisis/ilp_mcds_cyl.py` |
| `analisis/generar_figuras_pam_generation.py` | Generates all PAM paper figures: radar benchmark chart (Fig comparing real QKD deployments vs thesis networks) and Spain topology with degree/betweenness visual encoding | `python analisis/generar_figuras_pam_generation.py` |

**Results**: `datos/resultados_papers/tablas_pam_generation.csv` (Tables II–IV aggregate values)

---

### QKD-SKR-Routing (JOCN submission)

| Script | Description | Run |
|--------|-------------|-----|
| `analisis/enrutamiento_espana_completo.py` | All-pairs widest-path routing on the Spain 950-node PAM backbone (450,775 pairs); compares hop-minimal Dijkstra vs max-SKR bottleneck path; produces Tables 1–4 and Fig 2 | `python analisis/enrutamiento_espana_completo.py` |
| `analisis/delta_sensitivity_espana.py` | Δ-threshold sensitivity analysis (Table 5); rebuilds Spain graph from 950 PAM relay coordinates for each Δ ∈ {35, 40, 45, 50} km; consistent with PAM-generated backbone | `python analisis/delta_sensitivity_espana.py` |
| `analisis/generar_figuras_skr_routing.py` | Generates Fig 1 (SKR vs distance with calibrated BB84+decoy model, η_det=0.10), Fig 3 (Spain topology coloured by edge distance), Fig 4 (edge-distance and SKR distributions) | `python analisis/generar_figuras_skr_routing.py` |

**Results**: `datos/resultados_papers/tablas_skr_routing.json` (Tables 1–5 aggregate stats), `datos/resultados_papers/enrutamiento_espana_allpairs.csv` (450,775 pair-level routing results)

---

### Case III — Topological proxy derived from ADIF railway data

This case characterises a graph built from open railway-node and adjacency
records. It is not an inventory of deployable telecommunications links and
does not verify fibre continuity, optical loss or site availability.

| Script | Directory | Description |
|--------|-----------|-------------|
| `analisis_adif_junctions.py` | `adif/` | Main analysis pipeline: load ADIF CSVs → junction graph contraction (degree-2 chain reduction) → structural metrics → random failures (R=1000) → targeted attacks by degree and betweenness → JSON output |
| `generar_figura_adif.py` | `adif/` | Generates `adif_resiliencia.pdf/.png` — S(p) curves for both attack strategies |
| `generar_mapa_adif_static.py` | `adif/` | Generates georeferenced static map of junction graph (matplotlib); blue edges ≤50 km, orange >50 km, red nodes degree≥5 |
| `generar_mapa_adif.py` | `adif/` | Generates interactive HTML map (folium) — browser-only, not for LaTeX |
| `generar_matriz_adif.py` | `adif/` | Distance matrix between primary junctions |
| `generar_matriz_extendida.py` | `adif/` | Extended distance matrix with secondary junctions |

**Input files** (in `datos/adif/`):

| File | Description |
|------|-------------|
| `datos/adif/nodos_red_adif.csv` | ADIF node catalogue: 3,085 dependencies with geo-coordinates, category (P/S), connection status |
| `datos/adif/adyacencia_red_adif.csv` | ADIF railway adjacency records with reported lengths (km) |
| `datos/adif/resultados_adif_junctions.json` | Pre-computed analysis results: metrics, random failure statistics, attack curves, top-10 betweenness nodes |

**Key parameters (ADIF case):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `Δ_eff` | 50 km | Model classification assumption; not a calibrated operational threshold |
| `R` | 1,000 | Random failure replications |
| `p_0` | 13% | Fraction removed per trial |
| `p*` (degree attack) | 5% | Fragmentation threshold |
| `p*` (betweenness) | 10% | Fragmentation threshold |
| `\|V_J\|` | 485 | Junction nodes after degree-2 contraction |
| `\|E_J\|` | 633 | Edges in junction graph |
| Bridges | 138 | 21.8% of edges |
| Articulation points | 123 | 25.4% of nodes |

---

## How to run | Cómo ejecutar

### ADIF full pipeline

```bash
cd adif/
# 1. Build junction graph, compute all metrics and resilience curves
python analisis_adif_junctions.py
# → writes ../datos/adif/resultados_adif_junctions.json

# 2. Generate resilience figure
python generar_figura_adif.py
# → writes adif_resiliencia.pdf / .png

# 3. Generate static georeferenced map
python generar_mapa_adif_static.py
# → writes adif_junctions_mapa.pdf / .png
```

### CyL / España analysis pipeline

```bash
# Step 1: generate topology (R)
cd generacion/
Rscript medoids.R          # produces medoid assignments
Rscript components.R       # verifies connectivity
# → AdjacencyMatrixNamed45.csv (copy to working directory)

# Step 2: structural analysis (Python)
cd ../analisis/
python analis_redes_complejas.py
python metricas_avanzadas.py
python girvan_newmancyl.py          # default k=8; use `python girvan_newmancyl.py 10` for k=10
python k_core_decomposition.py
python enrutamiento_qkd.py
python benchmarks_qkd.py

# Step 3: resilience
cd ../ataques/
python ataques_aleatorios_nodos_fault.py
python ataques_dirigidos_nodos_fault.py
python ataques_dinamicos.py         # ~20-60 min for España (betweenness recomputed each step)
python ataques_aristas.py
python comparacion_resiliencia.py   # generates 3-case comparison figure with 95% CI bands

# Step 4: QKD physics
cd ../protocols/
python skr_bb84.py
```

---

## Reproducibility notes | Reproducibilidad

- **ADIF junction graph**: fully deterministic given the input CSVs (no random seed needed for the graph structure itself).
- **Random failures**: seed fixed via `random.seed(42)` + `numpy.random.seed(42)` in all simulation scripts.
- **PAM clustering**: seed fixed in `medoids.R` — see inline comment.
- **Girvan–Newman**: deterministic given the graph.
- **Louvain communities**: `seed=42` passed to `nx_comm.louvain_communities()`.
- **Shared utilities**: `qkd_utils.py` at repo root exposes `load_graph()`, `relative_gcc()`,
  `robustness_index()`, `p_star()`, `validate_adjacency_matrix()` and `get_thesis_style()`.
- **Thesis SKR/routing artefacts**: the source data, equations, tie-breaking rules,
  expected numerical checks and current provenance limitations are recorded in
  [`docs/THESIS_SKR_ROUTING_REPRODUCIBILITY.md`](docs/THESIS_SKR_ROUTING_REPRODUCIBILITY.md).
  Run `python -m unittest tests.test_thesis_skr_routing_audit` for an
  implementation-independent audit based only on the Python standard library.

---

## License and citation | Licencia y citación

**License:** [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html) — `SPDX-License-Identifier: GPL-3.0-only`

```
Copyright (C) 2019-2026 Iván García Cobo
```

<!-- After the first GitHub Release, replace the line below with the real Zenodo badge -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20582869.svg)](https://doi.org/10.5281/zenodo.20582869) -->

**Citation (software):**

If you use this toolkit, please cite it via Zenodo (DOI assigned on first release).
A machine-readable citation is available in [`CITATION.cff`](CITATION.cff).

**Citation (thesis):**
> García-Cobo, I. (2026). *Generación y validación de redes QKD a gran escala bajo restricciones físicas y tecnológicas*. Tesis doctoral, Universidad de Salamanca.

**Citation (doctoral journal article):**
> García-Cobo, I. & Menéndez, H.D. (2021). Designing large quantum key distribution networks via medoid-based algorithms. *Future Generation Computer Systems*, 115, 814–824. https://doi.org/10.1016/j.future.2020.09.037 
