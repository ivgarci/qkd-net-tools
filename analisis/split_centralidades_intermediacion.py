"""
Distribución de la centralidad de intermediación C_B(v) — CyL y España,
paneles individuales para la tesis.

NOTA (añadido 2026-07-29): la figura combinada Figures/centralidades_
intermediacion.pdf de la tesis (histograma de 2 paneles, "Frecuencia" en el
eje Y, anotaciones "Media = " / "Máx = " en español) no tiene un script
generador identificable en el historial de git de este repositorio — no es
el mismo script que ataques/figures_adversary_paper.py (decoy: 3 paneles,
etiquetas en inglés "Node count", para el envío a IEEE TDSC) ni que
adif/generar_centralidades_adif.py (decoy: paper ADIF-QKD). Los datos de
origen sí están confirmados con exactitud: la media, el máximo y el nodo
argmax de la columna "Betweenness Centrality" de
datos/cyl/Node_Specific_Network_Measures.csv y
datos/espana/Node_Specific_Network_Measures.csv reproducen exactamente
(a 4 decimales) los valores impresos en el PDF archivado
(CyL: media=0.0498, máx=0.1882 en Toro; España: media=0.0124, máx=0.1134
en Folgoso de la Ribera). Este script reconstruye los paneles a partir de
esos mismos CSV archivados, con el mismo estilo visual que el resto del
repositorio (ver analisis/coeficiente.py).

Genera:
  figuras/centralidades_intermediacion_cyl.pdf/.png
  figuras/centralidades_intermediacion_esp.pdf/.png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CYL = os.path.join(BASE, '..', 'datos', 'cyl')
DATA_ESP = os.path.join(BASE, '..', 'datos', 'espana')
FIGS_OUT = os.path.join(BASE, '..', 'figuras')
os.makedirs(FIGS_OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
})

CASES = [
    {
        'csv': os.path.join(DATA_CYL, 'Node_Specific_Network_Measures.csv'),
        'title': 'Castilla y León',
        'bins': 20,
        'stem': 'centralidades_intermediacion_cyl',
    },
    {
        'csv': os.path.join(DATA_ESP, 'Node_Specific_Network_Measures.csv'),
        'title': 'España peninsular',
        'bins': 30,
        'stem': 'centralidades_intermediacion_esp',
    },
]


def plot_case(case):
    df = pd.read_csv(case['csv'], index_col=0)
    cb = df['Betweenness Centrality']
    media = cb.mean()
    maximo = cb.max()
    nodo_max = cb.idxmax()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(cb.values, bins=case['bins'], color='steelblue',
            edgecolor='black', alpha=0.85)
    ax.axvline(media, color='firebrick', lw=1.6, ls='--')

    ax.text(0.97, 0.95, f'Media = {media:.4f}\nMáx = {maximo:.4f}\n({nodo_max})',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.7'))

    ax.set_title(case['title'])
    ax.set_xlabel(r'Centralidad de intermediación $C_B(v)$')
    ax.set_ylabel('Frecuencia')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(FIGS_OUT, f"{case['stem']}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")
    plt.close(fig)

    return media, maximo, nodo_max


if __name__ == '__main__':
    print("=" * 60)
    print("Centralidad de intermediación — paneles individuales")
    print("=" * 60)
    for case in CASES:
        media, maximo, nodo_max = plot_case(case)
        print(f"\n{case['title']}: media={media:.4f}, máx={maximo:.4f} ({nodo_max})")
    print("\nDone.")
