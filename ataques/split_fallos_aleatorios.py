"""
Distribución del tamaño relativo de la componente gigante S(G_p) bajo fallos
aleatorios de nodos (p0 = 13%) — CyL y España, paneles individuales para la
tesis.

NOTA (añadido 2026-07-29): la figura combinada Figures/resiliencia_fallos_
aleatorios.pdf de la tesis (histograma de 2 paneles, "Media = 0.84" / "Media
= 0.87") no tiene un script generador identificable en el historial de git
de este repositorio. NO es ataques/comparacion_resiliencia.py — ese script
genera figuras/comparacion_resiliencia_3casos.pdf, que es una figura
distinta de 3 paneles con curvas S(p) de ataques dirigidos por grado/
intermediación, sin histogramas de fallos aleatorios (ya descartado como
pista falsa antes de escribir este script). Los datos de origen sí están
confirmados con exactitud: la media, la desviación típica y el mínimo de
S_rel = LCC/|V| en datos/cyl/random_failure_results.csv (300 filas)
reproducen exactamente los valores citados en el capítulo 5 de la tesis
(media=0.842, sigma=0.037, mínimo=0.630; N_sim=300, p0=13%, semilla 42 fijada
en ataques/ataques_aleatorios_nodos_fault.py). El CSV de España
(datos/espana/random_failure_results.csv, 3000 filas, media=0.870) es el
equivalente para |V|=950, N_sim=3000; el script que lo generó no está en el
repositorio (solo el CSV fue archivado, commit d57db71), pero el recuento de
filas y el valor medio coinciden con el texto de la tesis
("N_sim = 3.000... Media = 0,87"). Este script reconstruye los paneles a
partir de esos mismos CSV archivados, con estilo visual consistente con el
resto del repositorio (ver analisis/coeficiente.py).

Genera:
  figuras/resiliencia_fallos_aleatorios_cyl.pdf/.png
  figuras/resiliencia_fallos_aleatorios_esp.pdf/.png
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
        'csv': os.path.join(DATA_CYL, 'random_failure_results.csv'),
        'N': 100,
        'title': r'Castilla y León ($|V|=100$, $p_0=13\%$, $R=300$)',
        'bins': 15,
        'stem': 'resiliencia_fallos_aleatorios_cyl',
    },
    {
        'csv': os.path.join(DATA_ESP, 'random_failure_results.csv'),
        'N': 950,
        'title': r'España peninsular ($|V|=950$, $p_0=13\%$, $R=3000$)',
        'bins': 30,
        'stem': 'resiliencia_fallos_aleatorios_esp',
    },
]


def plot_case(case):
    df = pd.read_csv(case['csv'])
    s_rel = df['Largest Connected Component Size'] / case['N']
    media = s_rel.mean()
    std = s_rel.std()
    minimo = s_rel.min()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(s_rel.values, bins=case['bins'], color='steelblue',
            edgecolor='black', alpha=0.85)
    ax.axvline(media, color='firebrick', lw=1.6, ls='--',
               label=f'Media = {media:.2f}')

    ax.set_title(case['title'])
    ax.set_xlabel(r'$S(G_p)$ — tamaño relativo de la componente gigante')
    ax.set_ylabel('Frecuencia')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Formatear eje X como porcentaje, igual que el PDF archivado
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
    )

    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(FIGS_OUT, f"{case['stem']}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")
    plt.close(fig)

    return media, std, minimo, len(df)


if __name__ == '__main__':
    print("=" * 60)
    print("Fallos aleatorios — paneles individuales")
    print("=" * 60)
    for case in CASES:
        media, std, minimo, n = plot_case(case)
        print(f"\n{case['title']}: N_sim={n}, media={media:.4f}, "
              f"std={std:.4f}, min={minimo:.4f}")
    print("\nDone.")
