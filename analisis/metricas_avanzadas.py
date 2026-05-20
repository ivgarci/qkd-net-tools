"""
Métricas avanzadas de redes complejas para los tres casos de estudio:
  - Eficiencia global E
  - Asortatividad r (correlación grado-grado)
  - Coeficiente small-world σ = (C/C_rand) / (L/L_rand)
  - Conectividad algebraica λ₂ (valor de Fiedler)
  - Exponente de escala libre α (ajuste log-log de P(k))

Genera datos/metricas_avanzadas_resumen.csv e imprime tabla para la tesis.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import linregress

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CYL  = os.path.join(BASE, '..', 'datos', 'cyl')
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
DATA_ADIF = os.path.join(BASE, '..', 'datos', 'adif')
OUT_DIR   = os.path.join(BASE, '..', 'datos')

SEED = 42


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_graph(adj_csv: str) -> nx.Graph:
    adj = pd.read_csv(adj_csv, index_col=0)
    return nx.from_pandas_adjacency(adj)


def load_adif_graph() -> nx.Graph:
    """Reconstruye el grafo de junctions ADIF desde el JSON pre-computado."""
    json_path = os.path.join(DATA_ADIF, 'resultados_adif_junctions.json')
    with open(json_path) as f:
        data = json.load(f)
    m = data['metrics']
    # El JSON no almacena la lista de aristas; usamos los datos de ataque
    # para verificar tamaño y devolvemos None si no se puede reconstruir.
    # En su lugar cargamos directamente los CSVs originales de ADIF.
    return None  # señal para usar métricas pre-computadas del JSON


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def global_efficiency(G: nx.Graph) -> float:
    """E = (1 / N(N-1)) * sum_{i≠j} 1/d(i,j). Usa NetworkX ≥3.0."""
    return nx.global_efficiency(G)


def small_world_sigma(G: nx.Graph, seed: int = SEED) -> float:
    """σ = (C/C_rand) / (L/L_rand). Requiere grafo conexo."""
    if not nx.is_connected(G):
        lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    else:
        lcc = G

    N = lcc.number_of_nodes()
    E = lcc.number_of_edges()
    C = nx.average_clustering(lcc)
    L = nx.average_shortest_path_length(lcc)

    # Grafo aleatorio de referencia con mismo N y E (Erdős–Rényi)
    p = 2 * E / (N * (N - 1))
    rng = np.random.default_rng(seed)
    C_rand_vals, L_rand_vals = [], []
    for _ in range(10):
        Gr = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(1e6)))
        if nx.is_connected(Gr):
            C_rand_vals.append(nx.average_clustering(Gr))
            L_rand_vals.append(nx.average_shortest_path_length(Gr))

    if not C_rand_vals:
        return float('nan')

    C_rand = np.mean(C_rand_vals)
    L_rand = np.mean(L_rand_vals)

    if C_rand == 0 or L_rand == 0:
        return float('nan')

    return round((C / C_rand) / (L / L_rand), 4)


def algebraic_connectivity(G: nx.Graph) -> float:
    """λ₂ de la matriz laplaciana (valor de Fiedler). Grafo no dirigido conexo."""
    if not nx.is_connected(G):
        lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    else:
        lcc = G
    return round(nx.algebraic_connectivity(lcc, seed=SEED), 6)


def scale_free_exponent(G: nx.Graph) -> tuple:
    """
    Ajuste lineal en log-log de P(k) vs k para estimar el exponente α.
    Devuelve (alpha, R²). alpha>0 indica cola de potencias.
    """
    degrees = [d for _, d in G.degree() if d > 0]
    if not degrees:
        return (float('nan'), float('nan'))

    k_vals, pk_vals = np.unique(degrees, return_counts=True)
    pk_vals = pk_vals / pk_vals.sum()

    # Filtrar k=0 y P(k)=0 para el log
    mask = (k_vals > 0) & (pk_vals > 0)
    log_k = np.log10(k_vals[mask].astype(float))
    log_pk = np.log10(pk_vals[mask].astype(float))

    if len(log_k) < 3:
        return (float('nan'), float('nan'))

    slope, intercept, r, p_val, se = linregress(log_k, log_pk)
    return (round(-slope, 3), round(r**2, 4))


def compute_advanced_metrics(G: nx.Graph, label: str) -> dict:
    print(f"\n  [{label}] Calculando métricas avanzadas...")

    N = G.number_of_nodes()
    E = G.number_of_edges()

    if not nx.is_connected(G):
        lcc_nodes = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(lcc_nodes).copy()
        print(f"    ⚠ Grafo no conexo — usando LCC ({len(lcc_nodes)} nodos)")
    else:
        G_lcc = G

    print(f"    Eficiencia global...", end=' ', flush=True)
    eff = round(global_efficiency(G_lcc), 6)
    print(eff)

    print(f"    Asortatividad...", end=' ', flush=True)
    asor = round(nx.degree_assortativity_coefficient(G), 6)
    print(asor)

    print(f"    Conectividad algebraica λ₂...", end=' ', flush=True)
    lam2 = algebraic_connectivity(G_lcc)
    print(lam2)

    print(f"    Exponente escala libre α...", end=' ', flush=True)
    alpha, r2 = scale_free_exponent(G)
    print(f"α={alpha}, R²={r2}")

    # small-world solo para grafos pequeños o medianos (evitar O(N²) en 950 nodos)
    if N <= 200:
        print(f"    Coeficiente small-world σ...", end=' ', flush=True)
        sigma = small_world_sigma(G_lcc)
        print(sigma)
    else:
        sigma = float('nan')
        print(f"    Coeficiente small-world σ: omitido (|V|={N} > 200, coste O(N²))")

    return {
        'Caso': label,
        '|V|': N,
        '|E|': E,
        'Eficiencia_global_E': eff,
        'Asortatividad_r': asor,
        'Small_world_sigma': sigma,
        'Conectividad_algebraica_lambda2': lam2,
        'Exponente_escala_libre_alpha': alpha,
        'Ajuste_R2': r2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 65)
    print("Métricas avanzadas de redes complejas — QKD Net Tools")
    print("=" * 65)

    resultados = []

    # Caso I: CyL
    G_cyl = load_graph(os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'))
    resultados.append(compute_advanced_metrics(G_cyl, 'CyL (|V|=100)'))

    # Caso II: España
    print("\n  [España] Cargando grafo 950 nodos (puede tardar ~1 min)...")
    G_esp = load_graph(os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv'))
    resultados.append(compute_advanced_metrics(G_esp, 'España (|V|=950)'))

    # Caso III: ADIF — métricas pre-computadas en JSON
    print("\n  [ADIF] Cargando métricas pre-computadas...")
    with open(os.path.join(DATA_ADIF, 'resultados_adif_junctions.json')) as f:
        adif_data = json.load(f)
    m = adif_data['metrics']

    # Para ADIF no tenemos el grafo en memoria; usamos los valores ya calculados
    # y completamos con N/A las métricas que requieren el grafo completo.
    adif_row = {
        'Caso': 'ADIF (|V|=485)',
        '|V|': m['V'],
        '|E|': m['E'],
        'Eficiencia_global_E': round(1.0 / m['mean_path'], 6) if m['mean_path'] else float('nan'),
        'Asortatividad_r': float('nan'),       # requiere grafo en memoria
        'Small_world_sigma': float('nan'),      # requiere grafo en memoria
        'Conectividad_algebraica_lambda2': float('nan'),
        'Exponente_escala_libre_alpha': float('nan'),
        'Ajuste_R2': float('nan'),
    }
    print("    (Para ADIF, eficiencia aproximada como 1/L_medio; resto requiere grafo en memoria)")
    resultados.append(adif_row)

    # Tabla final
    df = pd.DataFrame(resultados)

    print("\n" + "=" * 65)
    print("RESUMEN — Métricas avanzadas")
    print("=" * 65)
    cols_show = ['Caso', '|V|', '|E|', 'Eficiencia_global_E',
                 'Asortatividad_r', 'Small_world_sigma',
                 'Conectividad_algebraica_lambda2',
                 'Exponente_escala_libre_alpha', 'Ajuste_R2']
    with pd.option_context('display.float_format', '{:.4f}'.format,
                           'display.max_columns', 10,
                           'display.width', 120):
        print(df[cols_show].to_string(index=False))

    out_csv = os.path.join(OUT_DIR, 'metricas_avanzadas_resumen.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nGuardado: {out_csv}")
    print("\nDone.")
