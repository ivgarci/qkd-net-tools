# Intelligent networks for quantum key distribution — PhD research code

**Doctoral dissertation (official title, ES):** *Análisis de redes inteligentes para la distribución de claves cuánticas*  
**Working English title:** *Analysis of intelligent networks for quantum key distribution*

This repository collects **Python scripts** used in the thesis work: build graphs from adjacency matrices, compute network metrics (centrality, clustering, connectivity), detect communities (Louvain, Girvan–Newman), describe degree distributions, and simulate **resilience** under random failures and centrality-based targeted attacks. The empirical layer uses **named nodes** (e.g. places / regions) and, where coordinates exist, **geo-referenced** layouts.

> **Resumen (ES):** código de apoyo para la tesis *Análisis de redes inteligentes para la distribución de claves cuánticas*: análisis de grafos (métricas, comunidades, distribución de grado) y simulaciones de resiliencia ante fallos y ataques dirigidos, con visualizaciones y exportación de resultados a CSV y figuras.

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
| `ataques_aleatorios_nodos_fault.py` | Same as above | `random_failure_results.csv`, summary stats |
| `ataques_dirigidos_nodos_fault.py` | Same | `incremental_targeted_attack_results.csv`, disconnection threshold |
| `conectividad.py` | `adjacency_matrix.80.csv` | `node_connectivity_visualization.png` |
| `centralidad.py` | Graph **`G`** supplied by you | `centrality_measures_visualization.png` |
| `coeficiente.py` | **`G`** + complete imports (see below) | `clustering_coefficient_distribution.png` |
| `comunidad.py` | **`G`** + Louvain | `community_structure_visualization.png` |

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

**EN:** **Random removal** of 13% of nodes per trial, 300 trials; records largest connected component size, component count, and diameter of the largest component; writes `random_failure_results.csv` and descriptive statistics.

**ES:** Simulación Monte Carlo de fallos aleatorios (13 % de nodos); métricas de fragmentación y CSV.

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

**EN:** **Template:** `G` is not defined — assign a graph before running. Computes degree, betweenness, closeness; plots top-10 bar charts per metric.

**ES:** **Plantilla:** define `G` antes de ejecutar; barras de los 10 nodos con mayor centralidad.

---

### `coeficiente.py`

**EN:** **Snippet:** average and per-node clustering plus a histogram. **Missing** imports (`networkx`, `matplotlib.pyplot`, `numpy`) and graph `G`. Last line `plt.show(), avg_clustering_coefficient` should be cleaned up when you promote this to a full script.

**ES:** **Fragmento:** faltan imports y `G`; revisar la última línea al integrarlo.

---

### `comunidad.py`

**EN:** **Snippet:** requires predefined `G`. **Louvain** `best_partition`, community-coloured layout, community count.

**ES:** **Fragmento:** Louvain sobre `G` ya definido.

---

## Next steps | Próximos pasos sugeridos

**EN:** Centralise data paths (`data/`, env vars, or `config.yaml`); complete the three snippets with loaders and `if __name__ == "__main__":`; add minimal tests or a reproducibility notebook; fix or document the random **seed** for `ataques_aleatorios_nodos_fault.py` if you report confidence intervals.

**ES:** Unificar datos y rutas; completar plantillas; tests o notebook; semilla aleatoria documentada para intervalos de confianza.

---

## License and citation | Licencia y citación

**License | Licencia:** this repository is distributed under the [**GNU General Public License v3.0**](https://www.gnu.org/licenses/gpl-3.0.html) (GPL-3.0). Full legal text: [`LICENSE`](LICENSE).

`SPDX-License-Identifier: GPL-3.0-only`

**Copyright | Titularidad del copyright:**

```text
Copyright (C) 2020-2026 Iván García Cobo
```


