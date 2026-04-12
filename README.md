# Intelligent networks for quantum key distribution — PhD research code

**Doctoral dissertation (official title, ES):** *Análisis de redes inteligentes para la distribución de claves cuánticas*  
**Working English title:** *Analysis of intelligent networks for quantum key distribution*

This repository collects **Python scripts** used in the thesis work: build graphs from adjacency matrices (including **weighted** links if the CSV is non-binary), compute network metrics (centrality, clustering, connectivity, path length, global efficiency, assortativity, Louvain modularity), compare against **null models** (Erdős–Rényi `G(n,m)` and configuration), detect communities (Louvain, Girvan–Newman), describe degree distributions, and simulate **resilience** — random and targeted **node** attacks (static and **adaptive**), random and targeted **edge** attacks, full **S(p)** curves with Monte Carlo bands, and **Schneider-style R** summaries. Named nodes (e.g. places / regions) and **geo-referenced** layouts are used where coordinate CSVs exist.

> **Resumen (ES):** código para la tesis *Análisis de redes inteligentes para la distribución de claves cuánticas*: métricas extendidas, modularidad Louvain, modelos nulos, curvas de robustez S(p) para nodos y aristas (aleatorio, estático, adaptativo), métricas R, y scripts anteriores de visualización y ataques incrementales.

---

## Requirements | Requisitos

- **Python** 3.10+ recommended.
- Install dependencies:

```bash
pip install -r requirements.txt
```

| Package | Role |
|---------|------|
| `networkx` | Graphs and algorithms |
| `pandas` | CSV / adjacency matrices |
| `matplotlib` | Figures |
| `numpy` | Histograms (e.g. clustering) |
| `scikit-learn` | PCA, KMeans in `analis_redes_complejas.py` |
| `scipy` | ANOVA (`f_oneway`) |
| `python-louvain` | `comunidad.py` (`import community`) |

*Paquetes equivalentes en español:* mismos nombres (`pip install -r requirements.txt`).

---

## Expected input files | Datos de entrada

| File | Description |
|------|-------------|
| `AdjacencyMatrixNamed45.csv` | Symmetric adjacency matrix with **row and column names** as node IDs (used by several scripts). |
| `adjacency_matrix.80.csv` | Matrix **without** a header row (`header=None`) for `conectividad.py`. |
| `cyl_1000.csv` | Geographic coordinates: columns `Población`, `Longitud`, `Latitud` (separator `;`). |

**Note:** CSV files are **not** shipped with the repo; place them in the working directory or update paths inside each script.

*Nota (ES):* los CSV no se incluyen en el repositorio; colócalos en el directorio de ejecución o ajusta las rutas en cada script.

---

## Scripts and main outputs | Scripts y salidas

| Script | Main input | Main outputs |
|--------|------------|--------------|
| `analis_redes_complejas.py` | `AdjacencyMatrixNamed45.csv` | `hj_biplot_clusters.pdf`, `Node_Specific_Network_Measures.csv`, console stats |
| `generar_grado_distribucion.py` | `AdjacencyMatrixNamed45.csv` | `distribucion_grado_grafo.{png,pdf,svg}` |
| `girvan_newmancyl.py` | `AdjacencyMatrixNamed45.csv`, `cyl_1000.csv` | `girvan_newman_cyl_9.{png,pdf,svg}`, community listing |
| `ataques_aleatorios_nodos_fault.py` | Same as above | `random_failure_results.csv`, summary stats (`RNG_SEED` fixed for reproducibility) |
| `ataques_dirigidos_nodos_fault.py` | Same | `incremental_targeted_attack_results.csv`, disconnection threshold |
| `metricas_extendidas.py` | `AdjacencyMatrixNamed45.csv` | `extended_network_metrics.csv` |
| `modelos_nulos.py` | Same | `null_model_comparison.csv` (observed + random `G(n,m)` + configuration samples) |
| `robustez_avanzada.py` | Same (CLI flags) | `robustez_output/robustness_curves.csv`, `robustness_R_metrics.csv`, `robustness_nodes.png`, `robustness_edges.png` |
| `grafo_io.py` | — | Shared loader (`load_named_adjacency`); not run standalone |
| `conectividad.py` | `adjacency_matrix.80.csv` | `node_connectivity_visualization.png` |
| `centralidad.py` | `AdjacencyMatrixNamed45.csv` | `centrality_measures_visualization.png` |
| `coeficiente.py` | `AdjacencyMatrixNamed45.csv` | `clustering_coefficient_distribution.png` |
| `comunidad.py` | `AdjacencyMatrixNamed45.csv` | `community_structure_visualization.png` |

**Advanced robustness CLI (example):**

```bash
python robustez_avanzada.py --adjacency AdjacencyMatrixNamed45.csv --out-dir robustez_output --seed 42 --trials 80
```

Document `--seed` and `--trials` in the thesis when you report uncertainty bands.

---

## How to run | Cómo ejecutar

From the project root, with the required CSV files present (or after editing paths):

```bash
python analis_redes_complejas.py
python generar_grado_distribucion.py
# …
```

Scripts assume **relative paths** to the current working directory.

*En español:* ejecuta desde la raíz del proyecto con los CSV en la misma carpeta o cambiando rutas en el código.

---

## Code walkthrough | Explicación del código

### `analis_redes_complejas.py`

**EN:** Loads the named adjacency matrix, builds an undirected `networkx` graph, and computes degree / closeness / betweenness / eigenvector centrality (with non-convergence handling), per-node clustering, edge count, diameter (if connected), and density. Node-level measures are assembled in a DataFrame; **PCA** (2D, referred in-code as an HJ-Biplot-style view) and **KMeans** (3 clusters) follow, plus one-way **ANOVA** on degree centrality across clusters. Saves a PCA scatter coloured by cluster and a CSV of all measures and cluster labels.

**ES:** Carga la matriz nombrada, calcula centralidades, agrupamiento y propiedades globales; PCA + KMeans (3 grupos) + ANOVA sobre centralidad de grado; guarda figura y `Node_Specific_Network_Measures.csv`.

---

### `generar_grado_distribucion.py`

**EN:** Degree sequence histogram; figure exported to PNG, PDF, and SVG. (Plot title in the script still refers to a specific regional case — adjust for your chapter.)

**ES:** Histograma de la distribución de grado; exporta PNG, PDF y SVG.

---

### `girvan_newmancyl.py`

**EN:** Loads graph and node coordinates, runs **Girvan–Newman**, advances the partition generator, picks a multi-community split for colouring, draws the graph at lon/lat positions, exports figures, prints a community–nodes table.

**ES:** Girvan–Newman con posiciones geográficas y tablas de comunidades por consola.

---

### `ataques_aleatorios_nodos_fault.py`

**EN:** **Random removal** of 13% of nodes per trial, 300 trials; records largest connected component size, component count, and diameter of the largest component; writes `random_failure_results.csv` and descriptive statistics. Uses a fixed **`RNG_SEED`** (see script) for reproducibility.

**ES:** Monte Carlo al 13 %; CSV y estadísticos; semilla fija documentada.

---

### `metricas_extendidas.py`

**EN:** On the giant component: average shortest path length, diameter, **global efficiency** (supports weighted edges), average clustering, **degree assortativity**, **Louvain modularity** and community count. Writes one-row `extended_network_metrics.csv`.

**ES:** Métricas de camino/eficiencia, asortatividad, modularidad Louvain; CSV resumen.

---

### `modelos_nulos.py`

**EN:** Draws **Erdős–Rényi** `G(n,m)` graphs with the same `n` and `m` as the observed network, and **configuration-model** simple graphs from the observed degree sequence (multiple samples). Compares the same extended metrics to the **observed** graph; writes `null_model_comparison.csv`.

**ES:** Comparación con `G(n,m)` y configuración; CSV con observado + muestras.

---

### `robustez_avanzada.py`

**EN:** Builds **S(p)** curves: **random node** removal (mean ± std over trials), **static** and **adaptive** targeted removal by **degree, closeness, and betweenness**; **random edge** removal; **static** and **adaptive** removal by **edge betweenness**. Exports long-format `robustness_curves.csv`, aggregated **R_mean** (mean of S over the grid) and **R_integral** (trapezoidal ∫S(f)df) per scenario in `robustness_R_metrics.csv`, and two figures (`robustness_nodes.png`, `robustness_edges.png`). See module docstring for the Schneider-style interpretation.

**ES:** Curvas completas; nodos aleatorio + estático/adaptativo (grado, cercanía, intermediación); aristas; métricas R; CSV y figuras en `--out-dir`.

---

### `grafo_io.py`

**EN:** `load_named_adjacency(path)` — single entry point for the named-matrix CSV used across scripts.

**ES:** Carga unificada de la matriz de adyacencia nombrada.

---

### `ataques_dirigidos_nodos_fault.py`

**EN:** **Incremental targeted attack:** sort nodes by centrality (degree, closeness, or betweenness), remove 0–49% in 1% steps, track largest component, component count, diameter; saves CSV and reports the minimum removal fraction where the graph is no longer connected (`Number of Components > 1`).

**ES:** Ataque dirigido incremental por centralidad; umbral de desconexión y CSV.

---

### `conectividad.py`

**EN:** Loads headerless adjacency matrix, draws a labelled network sketch.

**ES:** Visualización básica de conectividad desde matriz sin cabecera.

---

### `centralidad.py`

**EN:** Loads the named adjacency graph; degree, betweenness, closeness; top-10 bar charts per metric.

**ES:** Carga la matriz nombrada; barras top-10 por centralidad.

---

### `coeficiente.py`

**EN:** Average and per-node clustering; histogram; saves PNG.

**ES:** Distribución del coeficiente de agrupamiento.

---

### `comunidad.py`

**EN:** **Louvain** `best_partition`, community-coloured layout, prints community count.

**ES:** Louvain y visualización por comunidades.

---

## Next steps | Próximos pasos sugeridos

**EN:** Centralise data paths (`data/`, env vars, or `config.yaml`); add a minimal reproducibility notebook; tie **edge weights** to QKD capacities in the CSV when the thesis model is ready.

**ES:** Rutas centralizadas; notebook; pesos QKD cuando el modelo lo defina.

---

## License and citation | Licencia y citación

**License | Licencia:** this repository is distributed under the [**GNU General Public License v3.0**](https://www.gnu.org/licenses/gpl-3.0.html) (GPL-3.0). Full legal text: [`LICENSE`](LICENSE).

`SPDX-License-Identifier: GPL-3.0-only`

**Copyright | Titularidad del copyright:**

```text
Copyright (C) 2020-2026 Iván García Cobo
```


