# Reproducibility record: thesis SKR and CyL routing results

This record concerns the two thesis artefacts:

- `Figures/skr_vs_distancia.pdf`
- `Figures/comparacion_rutas_qkd.pdf`

The PDF files currently present in the thesis repository were produced on
another computer. Their original scripts, environment lockfile, command line
and pair-level routing output were not preserved. A visually similar plot is
not evidence that its numerical claims were reproduced.

## Canonical inputs

The reproducible CyL calculation uses:

| Input | SHA-256 |
| --- | --- |
| `datos/cyl/AdjacencyMatrixNamed45.csv` | `430072428dedb25893468b581d416fd0a45bcd9655e68d085c0855e8952e52ef` |
| `datos/cyl/cyl_1000.csv` | `ab12f5d6de0c1e9773d3812604640e06e7b10658a5a6135a53b076f52e42a9eb` |

The adjacency matrix contains 100 nodes and 254 undirected edges. The
coordinate file covers every node in that matrix. Distances are great-circle
distances computed with radius 6371 km; they are not measured fibre lengths.

If either hash changes, numerical results must be recomputed and must not be
compared with the checks below as though the input were unchanged.

## Canonical idealised SKR model

The equations declared in Chapter 4 of the thesis are the canonical model for
these thesis figures. With

`alpha=0.2`, `eta_det=0.10`, `mu=0.5`, `Y0=1e-6`, `e0=0.5`,
`e_det=0.015`, `f_EC=1.16` and `q=0.5`, independent evaluation gives:

| Distance | SKR (bits per signal pulse) |
| ---: | ---: |
| 10 km | `6.4664382800499395e-3` |
| 50 km | `1.0188456107580379e-3` |
| 100 km | `9.996046924682088e-5` |
| 150 km | `8.253730896477398e-6` |

The `1e-8` reference cut is crossed between 190 and 191 km. It is a
numerical plotting/reference cut, not an operational service threshold.

This is an asymptotic, ideal single-photon-component estimate. It does not
implement finite-size bounds for a concrete multi-intensity decoy protocol.
Accordingly, no unused `mu_decoy` parameter should imply otherwise.

## Canonical routing definitions

All `binom(100,2) = 4,950` unordered CyL pairs must be included.

1. **Minimum-hop route:** minimise hop count; among equal-hop routes maximise
   bottleneck SKR.
2. **Max-min route:** maximise bottleneck SKR; among equal-bottleneck routes
   minimise hop count.
3. A final deterministic node-name ordering may select a concrete path when
   both metrics tie, but it must not change either reported metric.

Because the same strictly decreasing SKR function is applied to every edge,
maximising bottleneck SKR is equivalent here to minimising the greatest
great-circle edge distance in the path.

## Independently reproduced results

An implementation using only the standard library and the two canonical input
files obtains:

| Quantity | Result | Status of thesis claim |
| --- | ---: | --- |
| Unordered pairs | `4,950` | reproduced |
| Mean bottleneck-SKR ratio | `1.4417703475208163` | supports `1.442x` |
| Mean hop overhead | `3.993939393939394` | supports `3.994` |
| Minimum SKR on minimum-hop routes | `1.280838767155935e-3` | supports the stated approximate `1.28e-3` |
| Maximum pairwise improvement | `3.170653394445944` | reproducible diagnostic |

Both aggregate routing claims in the thesis are thus independently supported
by the public adjacency matrix, coordinates and the declared definitions.
The thesis PDF's routing figure labels the `1.44` mean ratio but not the
hop-overhead mean; the pair-level calculation, rather than the image alone,
provides the evidence for `3.994`.

The max-min calculation is performed in two phases. First, it finds the
optimal bottleneck. Second, it runs an unweighted shortest-path search in the
subgraph containing only edges whose SKR is at least that optimum. This second
phase is essential: ordering a single label per intermediate node by
`(bottleneck, hops)` does not implement the global hop tie-break correctly,
because two prefixes with different bottlenecks can become equal after a later
edge and the previously discarded, shorter prefix may then be preferable.

## Superseded historical outputs

Before the reproducibility repair, these repository outputs had been generated
by an older, different SKR formula:

- `datos/skr_per_link.csv`
- `datos/enrutamiento_qkd_bottleneck.csv` (only 300 sampled pairs)
- `figuras/skr_vs_distancia.pdf`
- `figuras/comparacion_rutas_qkd.pdf`

Those historical revisions must not be cited as reproducing the current
thesis text. The repaired pipeline additionally writes all 4,950 pair rows and
a machine-readable summary. Regeneration must remain atomic: script,
pair-level CSV, summary, PDF/PNG, dependency versions, input hashes and the Git
commit must describe the same run.

## Required release provenance

Before citing a release from the thesis, preserve:

- exact Git commit and a clean working tree;
- Python version and exact dependency versions;
- commands executed;
- SHA-256 of every input;
- all 4,950 pair-level CyL rows, not only a top-10 or a sample;
- a summary CSV computed from that CSV;
- PDF and PNG generated in the same run;
- the Git commit timestamp and the model/tie-break version. The generated
  artefacts deliberately omit wall-clock PDF metadata so that an unchanged
  input and environment produce byte-identical outputs.

Run the independent audit with:

```bash
python -m unittest tests.test_thesis_skr_routing_audit
```

The audit deliberately does not import NumPy, pandas, NetworkX or the
production scripts. This reduces the risk that a shared implementation error
makes both the result and its test agree.

The verification reported in this record used Python 3.14.6. The production
pipeline was additionally exercised with the exact versions in
`requirements-reproducibility.txt`. This narrow lock covers the repaired SKR
and CyL routing pipeline; `requirements.txt` remains the broader dependency
list for the rest of the repository.
