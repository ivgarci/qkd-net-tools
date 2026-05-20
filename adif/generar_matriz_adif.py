#!/usr/bin/env python3
"""
Build ADIF distance matrix between primary Renfe stations.

Enhanced graph: intermediate stations (not tramo endpoints) are inserted into the
graph via their COD_TRAMO field + geometric projection along the tramo LineString.
This gives full coverage, including stations like Madrid-Chamartín that are in the
tramos graph but missing from the Renfe CSV.

Primary nodes:
  - All tipo=E (Estación), estado=EX dependencias that appear as tramo endpoints
  - All Renfe AV/LD/MD CSV stations (Spain + Latour-de-Carol/Cerbère)
  Combined and deduplicated by COD_DEPEND code.
"""

import json
import csv
import heapq
import math
import os

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_ADIF  = os.path.join(BASE, '..', 'datos', 'adif')
DATA_RENFE = os.path.join(BASE, '..', 'datos', 'renfe')
FIGS_ADIF  = os.path.join(BASE, '..', 'figuras', 'adif')
os.makedirs(FIGS_ADIF, exist_ok=True)

# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def project_point_on_linestring(pt_lat, pt_lon, coords):
    """
    Project point (lat, lon) onto a LineString given as [[lon, lat], ...].
    Returns (cumulative_dist_m_from_start, total_line_length_m).
    """
    best_t = 0.0
    best_dist = float("inf")
    best_cum = 0.0
    cumulative = 0.0

    for i in range(len(coords) - 1):
        ax, ay = coords[i][0], coords[i][1]   # lon, lat
        bx, by = coords[i+1][0], coords[i+1][1]
        seg_len = haversine_m(ay, ax, by, bx)

        # Project pt onto segment [A, B] in planar coords (small scale OK)
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            t = 0.0
        else:
            t = ((pt_lon - ax) * dx + (pt_lat - ay) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))

        proj_lon = ax + t * dx
        proj_lat = ay + t * dy
        d = haversine_m(pt_lat, pt_lon, proj_lat, proj_lon)

        if d < best_dist:
            best_dist = d
            best_t = t
            best_cum = cumulative + t * seg_len

        cumulative += seg_len

    return best_cum, cumulative


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

print("Loading dependencias...")
with open(os.path.join(DATA_ADIF, "adif_dependencias.geojson")) as f:
    deps_features = json.load(f)["features"]

dep_info = {}
dep_coords = {}  # COD_DEPEND -> (lat, lon)
for feat in deps_features:
    p = feat["properties"]
    cod = p.get("COD_DEPEND", "")
    if not cod:
        continue
    dep_info[cod] = p
    coords = feat["geometry"]["coordinates"]  # [lon, lat]
    if coords:
        dep_coords[cod] = (coords[1], coords[0])

print(f"  {len(dep_info)} dependencias loaded")

print("Loading tramos...")
with open(os.path.join(DATA_ADIF, "adif_tramos.geojson")) as f:
    tramos_features = json.load(f)["features"]

tramo_by_cod = {}
for feat in tramos_features:
    cod = feat["properties"].get("CODTRAMO", "")
    if cod:
        tramo_by_cod[cod] = feat

print(f"  {len(tramo_by_cod)} tramos loaded")

# -------------------------------------------------------------------
# Build enhanced graph
# -------------------------------------------------------------------
# Base: tramo endpoints (DEPORIGEN, DEPDESTINO)
# Enhanced: insert intermediate dependencias using COD_TRAMO + geometry

print("Building enhanced graph...")

graph = {}  # {cod: [(neighbor, dist_m, tipo_red, linea), ...]}


def add_edge(src, dst, dist_m, tipo, linea):
    if src not in graph:
        graph[src] = []
    if dst not in graph:
        graph[dst] = []
    graph[src].append((dst, dist_m, tipo, linea))
    graph[dst].append((src, dist_m, tipo, linea))


# For each tramo, collect all intermediate stations that claim it via COD_TRAMO
tramo_intermediates = {}  # CODTRAMO -> list of (cod_depend, position_m)

for feat in deps_features:
    p = feat["properties"]
    cod = p.get("COD_DEPEND", "")
    ct = p.get("COD_TRAMO", "")
    if not cod or not ct or ct not in tramo_by_cod:
        continue
    # Don't re-process nodes that are already tramo endpoints
    tramo = tramo_by_cod[ct]
    tp = tramo["properties"]
    if cod == tp.get("DEPORIGEN") or cod == tp.get("DEPDESTINO"):
        continue
    if ct not in tramo_intermediates:
        tramo_intermediates[ct] = []
    tramo_intermediates[ct].append(cod)

# Build edges: base tramo edges + split tramos for intermediate stations
for feat in tramos_features:
    tp = feat["properties"]
    ct = tp.get("CODTRAMO", "")
    src = tp.get("DEPORIGEN", "")
    dst = tp.get("DEPDESTINO", "")
    total_len = tp.get("LONGITUD", 0) or 0
    tipo = tp.get("TIPO_RED", "")
    linea = tp.get("COD_LINEA", "")

    if not src or not dst:
        continue

    intermediates = tramo_intermediates.get(ct, [])

    if not intermediates:
        # Simple edge
        add_edge(src, dst, total_len, tipo, linea)
        continue

    # Project each intermediate station onto the tramo geometry
    line_coords = feat["geometry"]["coordinates"]  # [[lon, lat], ...]
    stations = [(src, 0.0)]  # (cod, position_m from src)

    _, line_total = project_point_on_linestring(0, 0, line_coords)  # just to measure total

    for inter_cod in intermediates:
        if inter_cod not in dep_coords:
            continue
        lat, lon = dep_coords[inter_cod]
        pos_m, _ = project_point_on_linestring(lat, lon, line_coords)
        stations.append((inter_cod, pos_m))

    stations.append((dst, total_len))
    # Sort by position along line
    stations.sort(key=lambda x: x[1])

    # Create edges between consecutive stations
    for i in range(len(stations) - 1):
        cod_a, pos_a = stations[i]
        cod_b, pos_b = stations[i + 1]
        seg_len = max(1, pos_b - pos_a)  # avoid 0-length
        add_edge(cod_a, cod_b, seg_len, tipo, linea)


print(f"  Graph nodes: {len(graph)}, edges (half): {sum(len(v) for v in graph.values())//2}")

# -------------------------------------------------------------------
# Primary nodes: Renfe CSV + all tipo=E, estado=EX tramo endpoints
# -------------------------------------------------------------------

# All tramo endpoint codes
tramo_endpoints = set()
for feat in tramos_features:
    tp = feat["properties"]
    if tp.get("DEPORIGEN"):
        tramo_endpoints.add(tp["DEPORIGEN"])
    if tp.get("DEPDESTINO"):
        tramo_endpoints.add(tp["DEPDESTINO"])

print("Building primary node list...")
primary = {}

# Tier 1: tipo=E, estado=EX stations that are tramo endpoints
for cod, p in dep_info.items():
    if p.get("COD_TIPO_D") == "E" and p.get("COD_ESTADO") == "EX" and cod in tramo_endpoints:
        if cod not in dep_coords:
            continue
        lat, lon = dep_coords[cod]
        primary[cod] = {
            "nombre": p.get("NOMBRE", ""),
            "lat": lat,
            "lon": lon,
            "provincia": "",
            "source": "adif_E",
        }

# Tier 2: Renfe AV/LD/MD CSV (Spain + border crossings)
with open(os.path.join(DATA_RENFE, "listado_completo_av_ld_md.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f, delimiter=";"):
        pais = row.get("PAIS", "").strip()
        code_key = [k for k in row.keys() if "DIGO" in k][0]
        code = row[code_key].strip()
        if pais == "España" or code in ("77310", "79316"):
            try:
                lat = float(row["LATITUD"].replace(",", "."))
                lon = float(row["LONGITUD"].replace(",", "."))
            except ValueError:
                continue
            if code not in primary:
                primary[code] = {
                    "nombre": row["DESCRIPCION"].strip(),
                    "lat": lat,
                    "lon": lon,
                    "provincia": row.get("PROVINCIA", "").strip(),
                    "source": "renfe_csv",
                }
            else:
                # Enrich with Renfe name/provincia if present
                primary[code]["provincia"] = row.get("PROVINCIA", "").strip()
                primary[code]["source"] = "both"

print(f"  Primary nodes: {len(primary)}")
primary_set = set(primary.keys())

# -------------------------------------------------------------------
# Dijkstra: find direct connections between primary nodes
# -------------------------------------------------------------------

def dijkstra_to_primaries(start, graph, primary_set):
    dist = {start: 0}
    connections = {}
    heap = [(0, start, "", "")]
    visited = set()

    while heap:
        d, node, tipo, linea = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)

        if node != start and node in primary_set:
            if node not in connections or d < connections[node][0]:
                connections[node] = (d, tipo, linea)
            continue  # don't expand past primary nodes

        if node not in graph:
            continue

        for neighbor, weight, edge_tipo, edge_linea in graph[node]:
            if neighbor in visited:
                continue
            new_dist = d + weight
            if neighbor not in dist or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor, edge_tipo, edge_linea))

    return connections


print(f"Computing connections for {len(primary)} primary nodes...")
print("(May take 2-3 minutes)")

results = []
seen_pairs = set()
processed = 0

for code, info in primary.items():
    if code not in graph:
        processed += 1
        continue

    conns = dijkstra_to_primaries(code, graph, primary_set)

    for nb_code, (dist_m, tipo, linea) in conns.items():
        pair = tuple(sorted([code, nb_code]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            dist_km = round(dist_m / 1000, 1)
            nb = primary[nb_code]
            results.append({
                "from_code": code,
                "from_nombre": info["nombre"],
                "from_provincia": info["provincia"],
                "to_code": nb_code,
                "to_nombre": nb["nombre"],
                "to_provincia": nb["provincia"],
                "dist_km": dist_km,
                "tipo_red": tipo,
                "linea": linea,
            })

    processed += 1
    if processed % 100 == 0:
        print(f"  {processed}/{len(primary)} done, {len(results)} connections...")

print(f"\nTotal direct connections: {len(results)}")

results.sort(key=lambda x: x["dist_km"])

# -------------------------------------------------------------------
# Isolated stations report
# -------------------------------------------------------------------
connected_codes = set()
for r in results:
    connected_codes.add(r["from_code"])
    connected_codes.add(r["to_code"])

isolated = {c: primary[c] for c in primary if c not in connected_codes}
print(f"Primary nodes with no connections: {len(isolated)}")

# -------------------------------------------------------------------
# Write CSV
# -------------------------------------------------------------------
csv_path = os.path.join(DATA_ADIF, "matriz_distancias_primarios.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "from_code", "from_nombre", "from_provincia",
        "to_code", "to_nombre", "to_provincia",
        "dist_km", "tipo_red", "linea"
    ])
    writer.writeheader()
    writer.writerows(results)
print(f"CSV saved: {csv_path}")

# -------------------------------------------------------------------
# Write HTML
# -------------------------------------------------------------------
n_stations = len(primary)
n_connected = len(connected_codes)
n_connections = len(results)
total_km = round(sum(r["dist_km"] for r in results), 0)
avg_km = round(total_km / n_connections, 1) if n_connections else 0

import json as _json
rows_json = _json.dumps(results)

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Matriz de Distancias ADIF — Nodos Primarios</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #111; color: #ccc; font-family: 'Segoe UI', monospace, sans-serif; font-size: 13px; }}

  #header {{
    background: #1a1a2e;
    border-bottom: 1px solid #333;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 30px;
    flex-wrap: wrap;
  }}
  #header h1 {{ color: #88aaff; font-size: 17px; font-weight: 600; }}
  #header h1 span {{ color: #556; font-size: 12px; font-weight: 400; display: block; margin-top: 2px; }}
  .stat {{ text-align: center; }}
  .stat .val {{ color: #88ffcc; font-size: 21px; font-weight: 700; }}
  .stat .lbl {{ color: #666; font-size: 11px; margin-top: 2px; }}

  #controls {{
    background: #161616;
    padding: 9px 20px;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    border-bottom: 1px solid #2a2a2a;
  }}
  #controls label {{ color: #888; font-size: 12px; }}
  #search {{
    background: #222; border: 1px solid #444; color: #eee;
    padding: 5px 10px; border-radius: 4px; width: 240px; font-size: 13px;
  }}
  #search:focus {{ outline: none; border-color: #66aaff; }}
  select {{
    background: #222; border: 1px solid #444; color: #eee;
    padding: 4px 8px; border-radius: 4px; font-size: 12px;
  }}
  #count {{ color: #666; font-size: 12px; margin-left: auto; }}

  #table-wrap {{ overflow: auto; height: calc(100vh - 130px); }}

  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  thead th {{
    background: #1e1e2e; color: #aabbdd; padding: 7px 10px;
    text-align: left; position: sticky; top: 0; cursor: pointer;
    user-select: none; white-space: nowrap; border-bottom: 2px solid #333;
  }}
  thead th:hover {{ background: #252540; }}
  thead th.asc::after {{ content: " ▲"; color: #88ffcc; }}
  thead th.desc::after {{ content: " ▼"; color: #88ffcc; }}

  tbody tr {{ border-bottom: 1px solid #1e1e1e; }}
  tbody tr:hover {{ background: #1a1a2a; }}
  tbody td {{ padding: 5px 10px; }}

  .code {{ color: #555; font-size: 11px; font-family: monospace; }}
  .nombre {{ color: #ddd; }}
  .prov {{ color: #777; font-size: 11px; }}
  .dist {{ color: #88ffcc; font-weight: 700; text-align: right; font-family: monospace; min-width: 55px; display: inline-block; }}
  .tipo {{ font-size: 11px; }}
  .t-av {{ color: #4488ff; }}
  .t-ic {{ color: #44aaff; }}
  .t-ic2 {{ color: #66bbdd; }}
  .t-cerc {{ color: #66cc66; }}
  .t-cerc2 {{ color: #44aa44; }}
  .t-merch {{ color: #CC8844; }}
  .t-resto {{ color: #AA6633; }}
  .linea {{ color: #555; font-size: 10px; max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
</style>
</head>
<body>
<div id="header">
  <h1>Matriz de Distancias ADIF<span>Conexiones directas entre nodos primarios · Estaciones tipo E (En explotación) + Renfe AV/LD/MD</span></h1>
  <div class="stat"><div class="val">{n_stations}</div><div class="lbl">Nodos primarios</div></div>
  <div class="stat"><div class="val">{n_connected}</div><div class="lbl">Con conexiones</div></div>
  <div class="stat"><div class="val">{n_connections:,}</div><div class="lbl">Conexiones directas</div></div>
  <div class="stat"><div class="val">{int(total_km):,} km</div><div class="lbl">Total red</div></div>
  <div class="stat"><div class="val">{avg_km} km</div><div class="lbl">Dist. media</div></div>
</div>

<div id="controls">
  <label>Buscar:</label>
  <input id="search" type="text" placeholder="Estación, código, provincia..." oninput="filterTable()">
  <label>Tipo red:</label>
  <select id="tipo-filter" onchange="filterTable()">
    <option value="">Todos</option>
    <option>Uso principal Alta Vel.</option>
    <option>Uso viaj. Interciud V&gt;160</option>
    <option>Uso resto viaj. Interciud</option>
    <option>N.Cercanías &gt;= 80 cir/día</option>
    <option>N.Cercanías &lt; 80 cir/día</option>
    <option>Uso mercancías</option>
    <option>Resto</option>
  </select>
  <span id="count"></span>
</div>

<div id="table-wrap">
<table id="main-table">
<thead>
  <tr>
    <th data-key="from_nombre">Origen</th>
    <th data-key="from_provincia">Prov.</th>
    <th data-key="to_nombre">Destino</th>
    <th data-key="to_provincia">Prov.</th>
    <th data-key="dist_km">km</th>
    <th data-key="tipo_red">Tipo red</th>
    <th>Línea</th>
  </tr>
</thead>
<tbody id="tbody"></tbody>
</table>
</div>

<script>
const ALL = {rows_json};
const TC = {{
  "Uso principal Alta Vel.":"t-av",
  "Uso viaj. Interciud V>160":"t-ic",
  "Uso resto viaj. Interciud":"t-ic2",
  "N.Cercanías >= 80 cir/día":"t-cerc",
  "N.Cercanías < 80 cir/día":"t-cerc2",
  "Uso mercancías":"t-merch",
  "Resto":"t-resto"
}};

let cur = [...ALL], sortKey = "dist_km", sortAsc = true;

document.querySelectorAll("thead th[data-key]").forEach(th => {{
  th.addEventListener("click", () => {{
    const k = th.dataset.key;
    if (sortKey === k) sortAsc = !sortAsc; else {{ sortKey = k; sortAsc = true; }}
    document.querySelectorAll("thead th").forEach(t => t.classList.remove("asc","desc"));
    th.classList.add(sortAsc ? "asc" : "desc");
    sortRender();
  }});
}});

function filterTable() {{
  const q = document.getElementById("search").value.toLowerCase();
  const tp = document.getElementById("tipo-filter").value;
  cur = ALL.filter(r => {{
    const mq = !q || [r.from_nombre,r.to_nombre,r.from_code,r.to_code,r.from_provincia,r.to_provincia].some(v=>v.toLowerCase().includes(q));
    const mt = !tp || r.tipo_red === tp;
    return mq && mt;
  }});
  sortRender();
}}

function sortRender() {{
  cur.sort((a,b) => {{
    let av=a[sortKey], bv=b[sortKey];
    if (typeof av==="number") return sortAsc?av-bv:bv-av;
    return sortAsc?String(av).localeCompare(String(bv)):String(bv).localeCompare(String(av));
  }});
  const tb = document.getElementById("tbody");
  const f = document.createDocumentFragment();
  const MAX = 2000;
  for (const r of cur.slice(0,MAX)) {{
    const tr = document.createElement("tr");
    const tc = TC[r.tipo_red]||"";
    tr.innerHTML = `
      <td><span class="nombre">${{r.from_nombre}}</span> <span class="code">${{r.from_code}}</span></td>
      <td class="prov">${{r.from_provincia}}</td>
      <td><span class="nombre">${{r.to_nombre}}</span> <span class="code">${{r.to_code}}</span></td>
      <td class="prov">${{r.to_provincia}}</td>
      <td><span class="dist">${{r.dist_km.toFixed(1)}}</span></td>
      <td class="tipo ${{tc}}">${{r.tipo_red}}</td>
      <td class="linea" title="${{r.linea}}">${{r.linea}}</td>`;
    f.appendChild(tr);
  }}
  tb.innerHTML = ""; tb.appendChild(f);
  document.getElementById("count").textContent =
    cur.length===ALL.length ? `${{ALL.length.toLocaleString()}} conexiones`
    : `${{cur.length.toLocaleString()}} de ${{ALL.length.toLocaleString()}} · mostrando ${{Math.min(MAX,cur.length).toLocaleString()}}`;
}}

// Init — sort by dist_km ascending
document.querySelector("th[data-key='dist_km']").classList.add("asc");
sortRender();
</script>
</body>
</html>
"""

html_path = os.path.join(FIGS_ADIF, "matriz_distancias_primarios.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"HTML saved: {html_path} ({round(len(html)/1024)}KB)")
print("\nTop 15 shortest connections:")
for r in results[:15]:
    print(f"  {r['from_nombre']:35s} — {r['to_nombre']:35s}: {r['dist_km']:6.1f} km  [{r['tipo_red']}]")

# Check Chamartin specifically
print("\nMadrid-Chamartín connections:")
for r in results:
    if '17000' in (r['from_code'], r['to_code']):
        other = r['to_nombre'] if r['from_code']=='17000' else r['from_nombre']
        print(f"  {other}: {r['dist_km']} km")
