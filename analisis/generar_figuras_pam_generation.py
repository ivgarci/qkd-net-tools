"""
Figures for QKD-PAM-Generation paper — unique replacements for shared figures.

Generates:
  QKD-PAM-Generation/Figures/benchmarks_qkd_metricas.pdf
      Radar (spider) chart comparing real-world QKD deployments and thesis
      networks across 6 normalised structural metrics.
  QKD-PAM-Generation/Figures/esp_topologia.png
      Geographic Spain QKD topology with nodes coloured by degree decile,
      sized by betweenness centrality — different visual from SKR-Routing version.

Run with:
    python analisis/generar_figuras_pam_generation.py
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from math import radians, sin, cos, sqrt, atan2

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_CYL  = os.path.join(BASE, '..', 'datos', 'cyl')
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
PAPER_DIR = os.path.join(BASE, '..', '..', '..', 'articulos',
                         'QKD-PAM-Generation', 'Figures')
os.makedirs(PAPER_DIR, exist_ok=True)


# ── reference QKD networks ────────────────────────────────────────────────────

def build_tokyo():
    G = nx.Graph(name='Tokyo QKD (2011)')
    nodes = {'Koganei': (35.700, 139.503), 'Otemachi': (35.686, 139.764),
             'Hakusan': (35.710, 139.752), 'Hongo': (35.712, 139.761),
             'Musashino': (35.706, 139.557), 'NICT': (35.706, 139.499)}
    for n, coords in nodes.items():
        G.add_node(n, pos=coords)
    for u, v, d in [('Koganei','NICT',1.0), ('Koganei','Musashino',7.0),
                    ('Musashino','Otemachi',45.0), ('Otemachi','Hakusan',2.0),
                    ('Hakusan','Hongo',1.0), ('Hongo','Otemachi',1.5),
                    ('Otemachi','NICT',45.0)]:
        G.add_edge(u, v, dist_km=d)
    return G


def build_secoqc():
    G = nx.Graph(name='SECOQC Vienna (2009)')
    for n in ['AIT','Brno','Siemens','T-Systems','Alcatel','ERG','CNET','Raiffeisen']:
        G.add_node(n)
    for u, v, d in [('AIT','Siemens',6.0), ('AIT','T-Systems',33.0),
                    ('AIT','Brno',200.0), ('Siemens','T-Systems',28.0),
                    ('Siemens','Alcatel',6.0), ('T-Systems','ERG',8.0),
                    ('Alcatel','ERG',12.0), ('Alcatel','CNET',5.0),
                    ('ERG','Raiffeisen',7.0), ('CNET','Raiffeisen',4.0),
                    ('T-Systems','CNET',20.0), ('Brno','ERG',200.0)]:
        G.add_edge(u, v, dist_km=d)
    return G


def build_china():
    G = nx.Graph(name='China Backbone (2021)')
    cities = ['Beijing','Jinan','Hefei','Shanghai','Wuhan','Guangzhou',
              'Chengdu','Xian','Zhengzhou','Nanjing','Hangzhou','Suzhou',
              'Tianjin','Qingdao','Changsha','Fuzhou','Nanchang','Guiyang',
              'Kunming','Chongqing','Harbin','Changchun','Shenyang','Dalian',
              'Hohhot','Taiyuan','Lanzhou','Urumqi','Lhasa','Shenzhen']
    for c in cities:
        G.add_node(c)
    for u, v, d in [
        ('Beijing','Tianjin',130), ('Tianjin','Jinan',350), ('Jinan','Zhengzhou',430),
        ('Zhengzhou','Wuhan',530), ('Wuhan','Changsha',340), ('Changsha','Guangzhou',680),
        ('Guangzhou','Shenzhen',140), ('Wuhan','Hefei',280), ('Hefei','Nanjing',150),
        ('Nanjing','Shanghai',300), ('Shanghai','Hangzhou',170), ('Hangzhou','Suzhou',120),
        ('Zhengzhou','Xian',510), ('Xian','Lanzhou',680), ('Lanzhou','Urumqi',1890),
        ('Xian','Chengdu',840), ('Chengdu','Chongqing',340), ('Chongqing','Guiyang',350),
        ('Guiyang','Guangzhou',780), ('Beijing','Hohhot',450), ('Beijing','Taiyuan',510),
        ('Beijing','Shenyang',700), ('Shenyang','Changchun',280), ('Changchun','Harbin',240),
        ('Shenyang','Dalian',390), ('Lanzhou','Lhasa',2400), ('Jinan','Qingdao',380),
        ('Shanghai','Fuzhou',800), ('Fuzhou','Nanchang',430), ('Nanchang','Wuhan',420),
    ]:
        if G.has_node(u) and G.has_node(v):
            G.add_edge(u, v, dist_km=d)
    return G


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(G, label):
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    N, E = G.number_of_nodes(), G.number_of_edges()
    degs = [d for _, d in G.degree()]
    metrics = {
        'label':      label,
        'V':          N,
        'E':          E,
        'density':    nx.density(G),
        'efficiency': nx.global_efficiency(G),
        'clustering': nx.average_clustering(G),
        'assort':     nx.degree_assortativity_coefficient(G),
        'cb_mean':    np.mean(list(nx.betweenness_centrality(G).values())),
    }
    try:
        metrics['lambda2'] = float(nx.algebraic_connectivity(G, seed=42))
    except Exception:
        metrics['lambda2'] = 0.0
    return metrics


# ── radar chart ───────────────────────────────────────────────────────────────

def radar_chart(df, metric_cols, metric_labels, out_path):
    n_metrics = len(metric_cols)
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    linestyles = ['-', '--', '-.', ':', '-']

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, row in df.iterrows():
        vals_raw = row[metric_cols].values.astype(float)
        # Normalise each metric to [0,1] across all networks
        col_mins = df[metric_cols].min().values
        col_maxs = df[metric_cols].max().values
        col_range = np.where(col_maxs - col_mins > 1e-12,
                             col_maxs - col_mins, 1.0)
        vals_norm = (vals_raw - col_mins) / col_range
        vals_norm = np.clip(vals_norm, 0, 1)
        vals_plot = vals_norm.tolist() + vals_norm[:1].tolist()
        color = palette[i % len(palette)]
        ax.plot(angles, vals_plot, lw=2, ls=linestyles[i % len(linestyles)],
                color=color, label=row['label'])
        ax.fill(angles, vals_plot, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=8, color='grey')
    ax.grid(True, alpha=0.35)
    ax.set_title('Structural metrics comparison\nreal-world QKD networks vs thesis networks',
                 size=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9.5)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        p = f'{out_path}.{ext}'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f'  Saved: {p}')
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


if __name__ == '__main__':
    # ── 1. Radar chart ────────────────────────────────────────────────────────
    print('Computing metrics for all networks...')
    builders = [
        ('Tokyo QKD (2011)',      build_tokyo),
        ('SECOQC Vienna (2009)',  build_secoqc),
        ('China Backbone (2021)', build_china),
    ]
    rows = []
    for label, fn in builders:
        G = fn()
        m = compute_metrics(G, label)
        rows.append(m)
        print(f'  {label}: |V|={m["V"]}, E_glob={m["efficiency"]:.3f}')

    for label, csv_name in [('CyL (|V|=100)', 'AdjacencyMatrixNamed45.csv'),
                             ('Spain (|V|=950)', 'AdjacencyMatrixNamed45.csv')]:
        data_dir = DATA_CYL if 'CyL' in label else DATA_ESP
        csv_path = os.path.join(data_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f'  Not found: {csv_path}')
            continue
        G = nx.from_pandas_adjacency(pd.read_csv(csv_path, index_col=0))
        m = compute_metrics(G, label)
        rows.append(m)
        print(f'  {label}: |V|={m["V"]}, E_glob={m["efficiency"]:.3f}')

    df = pd.DataFrame(rows)
    metric_cols   = ['efficiency', 'clustering', 'density', 'lambda2', 'cb_mean', 'assort']
    metric_labels = ['Global\nEfficiency', 'Clustering', 'Density',
                     r'$\lambda_2$', 'Mean $C_B$', 'Assortativity']

    out_bm = os.path.join(PAPER_DIR, 'benchmarks_qkd_metricas')
    radar_chart(df, metric_cols, metric_labels, out_bm)

    # ── 2. Spain topology — degree-coloured, betweenness-sized ────────────────
    print('Generating Spain topology figure...')
    adj_df = pd.read_csv(os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv'), index_col=0)
    G_esp  = nx.from_pandas_adjacency(adj_df)
    if not nx.is_connected(G_esp):
        G_esp = G_esp.subgraph(max(nx.connected_components(G_esp), key=len)).copy()

    coords_raw = pd.read_csv(os.path.join(DATA_ESP, 'peninsula_1000.csv'), sep=';', decimal=',')
    coords = {}
    for _, row in coords_raw.iterrows():
        name = str(row['Población']).strip()
        try:
            coords[name] = (float(str(row['Latitud']).replace(',', '.')),
                            float(str(row['Longitud']).replace(',', '.')))
        except (ValueError, KeyError):
            pass

    pos = {n: (coords[n][1], coords[n][0]) for n in G_esp.nodes() if n in coords}
    nodes_geo = list(pos.keys())

    if len(nodes_geo) >= 100:
        deg  = dict(G_esp.degree())
        cb   = nx.betweenness_centrality(G_esp, normalized=True)

        # Degree decile → colour
        all_degs   = np.array([deg[n] for n in nodes_geo])
        deg_pct    = np.percentile(all_degs, [20, 40, 60, 80])
        def deg_color(n):
            d = deg[n]
            if d <= deg_pct[0]: return '#d1e5f0'
            if d <= deg_pct[1]: return '#92c5de'
            if d <= deg_pct[2]: return '#4393c3'
            if d <= deg_pct[3]: return '#2166ac'
            return '#053061'

        node_colors = [deg_color(n) for n in nodes_geo]
        cb_vals     = np.array([cb.get(n, 0) for n in nodes_geo])
        node_sizes  = 5 + 80 * cb_vals / (cb_vals.max() + 1e-12)

        fig, ax = plt.subplots(figsize=(9, 7))
        nx.draw_networkx_edges(G_esp, pos, ax=ax, alpha=0.25,
                               edge_color='#999999', width=0.4)
        nx.draw_networkx_nodes(G_esp, pos, nodelist=nodes_geo, ax=ax,
                               node_color=node_colors, node_size=node_sizes,
                               alpha=0.9, edgecolors='white', linewidths=0.3)

        # Colourbar legend (degree deciles)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import BoundaryNorm, ListedColormap
        cmap = ListedColormap(['#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'])
        norm = BoundaryNorm([0, deg_pct[0], deg_pct[1], deg_pct[2], deg_pct[3], all_degs.max()+1], 5)
        sm   = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Node degree', fontsize=9)

        ax.set_xlabel('Longitude (°)', fontsize=10)
        ax.set_ylabel('Latitude (°)', fontsize=10)
        ax.set_title(f'Spain QKD relay backbone: $|V|={G_esp.number_of_nodes()}$ nodes, '
                     f'$|E|={G_esp.number_of_edges()}$ edges\n'
                     r'Node colour $\propto$ degree, size $\propto$ betweenness centrality',
                     fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        for ext in ('png', 'pdf'):
            out = os.path.join(PAPER_DIR, f'esp_topologia.{ext}')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f'  Saved: {out}')
        plt.close(fig)
    else:
        print('  Not enough coordinates for geographic plot — skipping.')

    # ── 3. CyL topology — same style as Spain ─────────────────────────────────
    print('Generating CyL topology figure...')
    adj_cyl = pd.read_csv(os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'), index_col=0)
    G_cyl   = nx.from_pandas_adjacency(adj_cyl)
    if not nx.is_connected(G_cyl):
        G_cyl = G_cyl.subgraph(max(nx.connected_components(G_cyl), key=len)).copy()

    coords_cyl_raw = pd.read_csv(os.path.join(DATA_CYL, 'cyl_1000.csv'), sep=';', decimal=',')
    coords_cyl = {}
    for _, row in coords_cyl_raw.iterrows():
        name = str(row['Población']).strip()
        try:
            coords_cyl[name] = (float(str(row['Latitud']).replace(',', '.')),
                                float(str(row['Longitud']).replace(',', '.')))
        except (ValueError, KeyError):
            pass

    pos_cyl   = {n: (coords_cyl[n][1], coords_cyl[n][0])
                 for n in G_cyl.nodes() if n in coords_cyl}
    nodes_cyl = list(pos_cyl.keys())
    print(f'  |V|={G_cyl.number_of_nodes()}, |E|={G_cyl.number_of_edges()}, '
          f'nodes with coords={len(nodes_cyl)}')

    if len(nodes_cyl) >= 20:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import BoundaryNorm, ListedColormap

        deg_cyl = dict(G_cyl.degree())
        cb_cyl  = nx.betweenness_centrality(G_cyl, normalized=True)

        all_degs_cyl = np.array([deg_cyl[n] for n in nodes_cyl])
        deg_pct_cyl  = np.percentile(all_degs_cyl, [20, 40, 60, 80])

        def deg_color_cyl(n):
            d = deg_cyl[n]
            if d <= deg_pct_cyl[0]: return '#d1e5f0'
            if d <= deg_pct_cyl[1]: return '#92c5de'
            if d <= deg_pct_cyl[2]: return '#4393c3'
            if d <= deg_pct_cyl[3]: return '#2166ac'
            return '#053061'

        node_colors_cyl = [deg_color_cyl(n) for n in nodes_cyl]
        cb_vals_cyl     = np.array([cb_cyl.get(n, 0) for n in nodes_cyl])
        node_sizes_cyl  = 12 + 200 * cb_vals_cyl / (cb_vals_cyl.max() + 1e-12)

        fig, ax = plt.subplots(figsize=(9, 7))
        nx.draw_networkx_edges(G_cyl, pos_cyl, ax=ax, alpha=0.30,
                               edge_color='#999999', width=0.5)
        nx.draw_networkx_nodes(G_cyl, pos_cyl, nodelist=nodes_cyl, ax=ax,
                               node_color=node_colors_cyl, node_size=node_sizes_cyl,
                               alpha=0.92, edgecolors='white', linewidths=0.4)

        # Labels for highest-betweenness nodes
        top_cb = sorted(nodes_cyl, key=lambda n: cb_cyl.get(n, 0), reverse=True)[:8]
        labels_cyl = {n: n for n in top_cb}
        nx.draw_networkx_labels(G_cyl, pos_cyl, labels=labels_cyl, ax=ax,
                                font_size=6, font_color='#222222',
                                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                          alpha=0.65, ec='none'))

        cmap_cyl = ListedColormap(['#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'])
        norm_cyl = BoundaryNorm(
            [0, deg_pct_cyl[0], deg_pct_cyl[1], deg_pct_cyl[2],
             deg_pct_cyl[3], all_degs_cyl.max() + 1], 5)
        sm_cyl = ScalarMappable(cmap=cmap_cyl, norm=norm_cyl)
        sm_cyl.set_array([])
        cbar_cyl = fig.colorbar(sm_cyl, ax=ax, fraction=0.025, pad=0.02)
        cbar_cyl.set_label('Node degree', fontsize=9)

        ax.set_xlabel('Longitude (°)', fontsize=10)
        ax.set_ylabel('Latitude (°)', fontsize=10)
        ax.set_title(f'CyL QKD relay backbone: $|V|={G_cyl.number_of_nodes()}$ nodes, '
                     f'$|E|={G_cyl.number_of_edges()}$ edges\n'
                     r'Node colour $\propto$ degree, size $\propto$ betweenness centrality',
                     fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        for ext in ('png', 'pdf'):
            out = os.path.join(PAPER_DIR, f'cyl_topologia.{ext}')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f'  Saved: {out}')
        plt.close(fig)
    else:
        print('  Not enough coordinates for CyL plot — skipping.')

    print('Done.')
