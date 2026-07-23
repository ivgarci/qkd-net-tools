# Routing scenarios and canonical outputs

All routing analyses use `analisis/routing_core.py` and the ideal asymptotic
BB84 model in `protocols/skr_bb84.py`. The two policies are:

1. minimum hops, then maximum SKR bottleneck;
2. maximum SKR bottleneck, then minimum hops.

Residual path ties use a deterministic node-name order. The fast metrics-only
engine is checked against the exhaustive Pareto-path implementation on small
and random graphs.

## Canonical graph snapshots

| Case | Nodes | Edges | Unordered pairs | Distance interpretation |
| --- | ---: | ---: | ---: | --- |
| CyL | 100 | 254 | 4,950 | Haversine, factor 1.0 |
| Peninsular Spain | 950 | 5,681 | 450,775 | Haversine, factor 1.0 |
| ADIF proxy LCC | 2,735 | 2,910 | 3,738,745 | Reported railway length; scenario proxy only |

The current canonical aggregate checks are:

| Case | Mean SKR gain | Mean hop overhead |
| --- | ---: | ---: |
| CyL | `1.4417703475208163` | `3.993939393939394` |
| Peninsular Spain | `2.179604963197868` | `30.061651600022184` |
| ADIF proxy LCC | `2.8593050667271256` | `77.46515287883797` |

The ADIF calculation is not evidence of dark-fibre availability or optical
feasibility. It only applies the declared model to a railway-derived proxy.

## Explicit non-canonical scenarios

`--distance-factor 1.25` is retained for sensitivity or paper comparison, but
it is not an experimentally established fibre-routing factor. On the archived
Spain graph it changes SKR values and gain ratios, not the selected paths or
hop overhead while SKR remains strictly decreasing with distance.

`analisis/delta_sensitivity_espana.py` reconstructs new geometric graphs. They
are not interchangeable with the archived 5,681-edge snapshot. In particular,
reconstruction at 45 km yields 5,690 edges: 14 additions and 5 removals. The
generated JSON records the exact edge-set hashes and samples.

## Commands and outputs

```bash
python analisis/enrutamiento_qkd.py --case cyl
python analisis/enrutamiento_qkd.py --case espana
python analisis/enrutamiento_espana_completo.py
python analisis/delta_sensitivity_espana.py
python analisis/enrutamiento_adif_completo.py
```

Each case writes a separate summary. Spain's canonical pair-level CSV and
figures are:

- `datos/resultados_papers/enrutamiento_espana_allpairs.csv`
- `datos/resultados_papers/enrutamiento_espana_summary.csv`
- `figuras/comparacion_rutas_qkd_espana.pdf`
- `figuras/comparacion_rutas_qkd_espana.png`

The former mixed `tablas_skr_routing.json` was removed because it combined
incompatible models, placeholders and stale values.

The old `null_model_er*.csv` artefacts were also removed after the SKR repair:
regenerating the complete 50-realisation Spain experiment is intentionally
left as an explicit long-running release step rather than presenting outputs
from the superseded physical model as current results.
