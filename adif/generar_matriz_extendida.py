#!/usr/bin/env python3
"""
Matriz extendida: conexiones directas entre todos los nodos conectados.

Nodos primarios   (P): estaciones de todos los CSV Renfe/FEVE/catálogo — 1182 nodos
Nodos secundarios (S): dependencias ADIF conectadas no presentes en CSVs — 1578 nodos

Output:
  matriz_extendida.csv
  matriz_extendida.html
"""

import json, csv, heapq, math, os

BASE  = "/Users/igarcia/doctorado/2025_2026/mapas"
ADIF  = os.path.join(BASE, "adif")
RENFE = os.path.join(BASE, "renfe")

# ── Utilidades ────────────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def project_point_on_linestring(pt_lat, pt_lon, coords):
    best_dist = float("inf")
    best_cum  = 0.0
    cumulative = 0.0
    for i in range(len(coords) - 1):
        ax, ay = coords[i][0],   coords[i][1]
        bx, by = coords[i+1][0], coords[i+1][1]
        seg_len = haversine_m(ay, ax, by, bx)
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            t = 0.0
        else:
            t = max(0.0, min(1.0,
                ((pt_lon - ax)*dx + (pt_lat - ay)*dy) / (dx*dx + dy*dy)))
        d = haversine_m(pt_lat, pt_lon, ay + t*dy, ax + t*dx)
        if d < best_dist:
            best_dist = d
            best_cum  = cumulative + t * seg_len
        cumulative += seg_len
    return best_cum, cumulative

# ── Cargar dependencias ───────────────────────────────────────────────────────

print("Cargando dependencias...")
with open(os.path.join(ADIF, "adif_dependencias.geojson")) as f:
    deps_features = json.load(f)["features"]

dep_info   = {}   # cod -> properties
dep_coords = {}   # cod -> (lat, lon)
for feat in deps_features:
    p   = feat["properties"]
    cod = p.get("COD_DEPEND", "")
    g   = feat.get("geometry") or {}
    c   = g.get("coordinates")
    if cod and c:
        dep_info[cod]   = p
        dep_coords[cod] = (c[1], c[0])

# ── Cargar tramos ─────────────────────────────────────────────────────────────

print("Cargando tramos...")
with open(os.path.join(ADIF, "adif_tramos.geojson")) as f:
    tramos_features = json.load(f)["features"]

tramo_by_cod = {f["properties"]["CODTRAMO"]: f for f in tramos_features
                if f["properties"].get("CODTRAMO")}

# ── Calcular nodos conectados ─────────────────────────────────────────────────

tramo_by_codtramo_set = set(tramo_by_cod.keys())
tramo_endpoints = set()
for feat in tramos_features:
    p = feat["properties"]
    if p.get("DEPORIGEN"):  tramo_endpoints.add(p["DEPORIGEN"])
    if p.get("DEPDESTINO"): tramo_endpoints.add(p["DEPDESTINO"])

dep_linked = set()
for feat in deps_features:
    p   = feat["properties"]
    cod = p.get("COD_DEPEND","")
    ct  = p.get("COD_TRAMO","")
    if cod and ct and ct in tramo_by_codtramo_set:
        dep_linked.add(cod)

CONNECTED = tramo_endpoints | dep_linked
print(f"  Nodos conectados: {len(CONNECTED)}")

# ── Construir grafo mejorado (con estaciones intermedias) ─────────────────────

print("Construyendo grafo...")
graph = {}

def add_edge(src, dst, dist_m, tipo, linea):
    if not src or not dst: return
    graph.setdefault(src, []).append((dst, dist_m, tipo, linea))
    graph.setdefault(dst, []).append((src, dist_m, tipo, linea))

# Agrupar intermedias por tramo
tramo_intermediates = {}
for feat in deps_features:
    p   = feat["properties"]
    cod = p.get("COD_DEPEND","")
    ct  = p.get("COD_TRAMO","")
    if not cod or not ct or ct not in tramo_by_cod:
        continue
    tp = tramo_by_cod[ct]["properties"]
    if cod in (tp.get("DEPORIGEN"), tp.get("DEPDESTINO")):
        continue
    tramo_intermediates.setdefault(ct, []).append(cod)

for feat in tramos_features:
    tp       = feat["properties"]
    ct       = tp.get("CODTRAMO","")
    src      = tp.get("DEPORIGEN","")
    dst      = tp.get("DEPDESTINO","")
    total_len = tp.get("LONGITUD", 0) or 0
    tipo     = tp.get("TIPO_RED","")
    linea    = tp.get("COD_LINEA","")
    if not src or not dst: continue

    intermediates = tramo_intermediates.get(ct, [])
    if not intermediates:
        add_edge(src, dst, total_len, tipo, linea)
        continue

    line_coords = feat["geometry"]["coordinates"]
    stations = [(src, 0.0)]
    for inter in intermediates:
        if inter not in dep_coords: continue
        lat, lon = dep_coords[inter]
        pos_m, _ = project_point_on_linestring(lat, lon, line_coords)
        stations.append((inter, pos_m))
    stations.append((dst, total_len))
    stations.sort(key=lambda x: x[1])
    for i in range(len(stations) - 1):
        ca, pa = stations[i]
        cb, pb = stations[i+1]
        add_edge(ca, cb, max(1, pb - pa), tipo, linea)

print(f"  Grafo: {len(graph)} nodos, {sum(len(v) for v in graph.values())//2} aristas")

# ── Cargar nodo primario / secundario ─────────────────────────────────────────

print("Clasificando nodos...")
BORDER = {"77310","79316"}
station_codes = set()

# Catálogo completo
with open(os.path.join(RENFE, "estaciones.csv"), encoding="latin-1") as f:
    for row in csv.DictReader(f, delimiter=";"):
        if row.get("PAIS","").strip().upper() in ("ESPAÑA","ESPA\xd1A"):
            station_codes.add(row["CODIGO"].strip().strip('"'))

# FEVE
with open(os.path.join(RENFE, "listado-de-estaciones-feve-2.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f, delimiter=";"):
        ck = [k for k in row.keys() if "DIGO" in k][0]
        station_codes.add(row[ck].strip())

# AV/LD/MD
with open(os.path.join(RENFE, "listado_completo_av_ld_md.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f, delimiter=";"):
        pais = row.get("PAIS","").strip()
        ck   = [k for k in row.keys() if "DIGO" in k][0]
        code = row[ck].strip()
        if pais == "España" or code in BORDER:
            station_codes.add(code)

# Nodos de interés = todos los CONNECTED (en grafo)
interest_nodes = {}   # cod -> {nombre, lat, lon, categoria, tipo_dep}
TIPO_D_LABEL = {
    "E":"Estación","P":"Punto KM","B":"Bifurcación","F":"Fin de línea",
    "T":"Túnel","Y":"Apartadero","A":"Apeadero","K":"Cargadero",
    "G":"Garita","O":"Otro","Z":"Zona técnica","C":"Cruce","D":"Depósito",
}

for cod in CONNECTED:
    if cod not in dep_info: continue
    p    = dep_info[cod]
    lat, lon = dep_coords.get(cod, (None, None))
    if lat is None: continue
    cat = "P" if cod in station_codes else "S"
    interest_nodes[cod] = {
        "cod":      cod,
        "nombre":   p.get("NOMBRE",""),
        "lat":      lat,
        "lon":      lon,
        "categoria": cat,
        "tipo_dep": TIPO_D_LABEL.get(p.get("COD_TIPO_D",""), p.get("COD_TIPO_D","")),
        "estado":   p.get("COD_ESTADO",""),
        "titular":  p.get("COD_TITULA",""),
    }

n_primary   = sum(1 for n in interest_nodes.values() if n["categoria"] == "P")
n_secondary = sum(1 for n in interest_nodes.values() if n["categoria"] == "S")
print(f"  Nodos primarios (P): {n_primary}  |  Secundarios (S): {n_secondary}")

interest_set = set(interest_nodes.keys())

# ── Dijkstra desde cada nodo de interés ──────────────────────────────────────

def dijkstra_to_interest(start, graph, interest_set):
    dist    = {start: 0}
    conns   = {}
    heap    = [(0, start, "", "")]
    visited = set()
    while heap:
        d, node, tipo, linea = heapq.heappop(heap)
        if node in visited: continue
        visited.add(node)
        if node != start and node in interest_set:
            if node not in conns or d < conns[node][0]:
                conns[node] = (d, tipo, linea)
            continue
        for nb, w, et, el in graph.get(node, []):
            if nb in visited: continue
            nd = d + w
            if nb not in dist or nd < dist[nb]:
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb, et, el))
    return conns

print(f"Calculando conexiones para {len(interest_nodes)} nodos...")
print("(Puede tardar 3-5 minutos)")

results    = []
seen_pairs = set()
processed  = 0

for cod, info in interest_nodes.items():
    if cod not in graph:
        processed += 1
        continue
    conns = dijkstra_to_interest(cod, graph, interest_set)
    for nb_cod, (dist_m, tipo, linea) in conns.items():
        pair = tuple(sorted([cod, nb_cod]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            nb = interest_nodes[nb_cod]
            # Tipo de conexión: PP / PS / SS
            cat_pair = "".join(sorted([info["categoria"], nb["categoria"]]))
            results.append({
                "from_cod":      cod,
                "from_nombre":   info["nombre"],
                "from_cat":      info["categoria"],
                "from_tipo_dep": info["tipo_dep"],
                "to_cod":        nb_cod,
                "to_nombre":     nb["nombre"],
                "to_cat":        nb["categoria"],
                "to_tipo_dep":   nb["tipo_dep"],
                "dist_km":       round(dist_m / 1000, 1),
                "tipo_red":      tipo,
                "linea":         linea,
                "cat_par":       cat_pair,   # PP / PS / SS
            })
    processed += 1
    if processed % 300 == 0:
        print(f"  {processed}/{len(interest_nodes)} procesados, {len(results)} conexiones...")

print(f"\nTotal conexiones: {len(results)}")
results.sort(key=lambda x: x["dist_km"])

# Stats
from collections import Counter
cat_counts = Counter(r["cat_par"] for r in results)
print(f"  PP (primario-primario):    {cat_counts['PP']}")
print(f"  PS (primario-secundario):  {cat_counts['PS']}")
print(f"  SS (secundario-secundario):{cat_counts['SS']}")

# ── CSV ───────────────────────────────────────────────────────────────────────

csv_path = os.path.join(BASE, "matriz_extendida.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "from_cod","from_nombre","from_cat","from_tipo_dep",
        "to_cod","to_nombre","to_cat","to_tipo_dep",
        "dist_km","tipo_red","linea","cat_par"
    ])
    writer.writeheader()
    writer.writerows(results)
print(f"CSV: {csv_path}")

# ── HTML ──────────────────────────────────────────────────────────────────────

import json as _json
rows_json = _json.dumps(results)

n_conn  = len(results)
tot_km  = round(sum(r["dist_km"] for r in results))
avg_km  = round(tot_km / n_conn, 1) if n_conn else 0

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Matriz Extendida ADIF</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ background:#111; color:#ccc; font-family:'Segoe UI',monospace,sans-serif; font-size:13px }}

#header {{
  background:#1a1a2e; border-bottom:1px solid #333;
  padding:13px 20px; display:flex; align-items:center; gap:28px; flex-wrap:wrap;
}}
#header h1 {{ color:#88aaff; font-size:17px; font-weight:600 }}
#header h1 span {{ color:#445; font-size:12px; font-weight:400; display:block; margin-top:2px }}
.stat .val {{ color:#88ffcc; font-size:20px; font-weight:700 }}
.stat .lbl {{ color:#555; font-size:11px; margin-top:2px }}

#controls {{
  background:#161616; padding:8px 20px; display:flex;
  gap:12px; align-items:center; flex-wrap:wrap; border-bottom:1px solid #222;
}}
#controls label {{ color:#888; font-size:12px }}
#search {{ background:#222; border:1px solid #444; color:#eee; padding:5px 10px;
           border-radius:4px; width:230px; font-size:13px }}
#search:focus {{ outline:none; border-color:#66aaff }}
select {{ background:#222; border:1px solid #444; color:#eee; padding:4px 8px; border-radius:4px; font-size:12px }}
#count {{ color:#555; font-size:12px; margin-left:auto }}

#table-wrap {{ overflow:auto; height:calc(100vh - 128px) }}
table {{ width:100%; border-collapse:collapse; font-size:12px }}
thead th {{
  background:#1e1e2e; color:#aabbdd; padding:7px 9px; text-align:left;
  position:sticky; top:0; cursor:pointer; user-select:none; white-space:nowrap;
  border-bottom:2px solid #333;
}}
thead th:hover {{ background:#252540 }}
thead th.asc::after  {{ content:" ▲"; color:#88ffcc }}
thead th.desc::after {{ content:" ▼"; color:#88ffcc }}
tbody tr {{ border-bottom:1px solid #1a1a1a }}
tbody tr:hover {{ background:#1a1a2a }}
tbody td {{ padding:5px 9px }}

/* categoria badges */
.cat-P {{ background:#1a3a1a; color:#88ee88; border-radius:3px; padding:1px 5px; font-size:10px; font-weight:600 }}
.cat-S {{ background:#1a1a3a; color:#8888ee; border-radius:3px; padding:1px 5px; font-size:10px; font-weight:600 }}

.nombre {{ color:#ddd }}
.cod {{ color:#555; font-size:11px; font-family:monospace }}
.tipo_dep {{ color:#666; font-size:10px }}
.dist {{ color:#88ffcc; font-weight:700; font-family:monospace; text-align:right }}

/* tipo red colors */
.t-av   {{ color:#4488ff }}
.t-ic   {{ color:#44aaff }}
.t-ic2  {{ color:#66bbdd }}
.t-cerc {{ color:#66cc66 }}
.t-cerc2{{ color:#44aa44 }}
.t-merch{{ color:#CC8844 }}
.t-resto{{ color:#AA6633 }}

.linea {{ color:#444; font-size:10px; max-width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap }}

/* cat-par badge */
.pp {{ color:#88ee88; font-size:10px; font-weight:700 }}
.ps {{ color:#eeee88; font-size:10px; font-weight:700 }}
.ss {{ color:#8888ee; font-size:10px; font-weight:700 }}
</style>
</head>
<body>

<div id="header">
  <h1>Matriz Extendida ADIF<span>Conexiones directas · Estaciones (P) + Dependencias ADIF (S)</span></h1>
  <div class="stat"><div class="val">{n_primary}</div><div class="lbl">Nodos P (estaciones)</div></div>
  <div class="stat"><div class="val">{n_secondary}</div><div class="lbl">Nodos S (dependencias)</div></div>
  <div class="stat"><div class="val">{n_conn:,}</div><div class="lbl">Conexiones directas</div></div>
  <div class="stat"><div class="val">{cat_counts['PP']:,}</div><div class="lbl">P↔P</div></div>
  <div class="stat"><div class="val">{cat_counts['PS']:,}</div><div class="lbl">P↔S</div></div>
  <div class="stat"><div class="val">{cat_counts['SS']:,}</div><div class="lbl">S↔S</div></div>
  <div class="stat"><div class="val">{avg_km} km</div><div class="lbl">Dist. media</div></div>
</div>

<div id="controls">
  <label>Buscar:</label>
  <input id="search" type="text" placeholder="Nombre, código..." oninput="filter()">
  <label>Tipo conexión:</label>
  <select id="cat-filter" onchange="filter()">
    <option value="">Todos</option>
    <option value="PP">P↔P (estación–estación)</option>
    <option value="PS">P↔S (estación–dependencia)</option>
    <option value="SS">S↔S (dependencia–dependencia)</option>
  </select>
  <label>Tipo red:</label>
  <select id="tipo-filter" onchange="filter()">
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
<table>
<thead>
  <tr>
    <th data-key="from_nombre">Origen</th>
    <th data-key="from_tipo_dep">Tipo</th>
    <th data-key="to_nombre">Destino</th>
    <th data-key="to_tipo_dep">Tipo</th>
    <th data-key="dist_km">km</th>
    <th data-key="cat_par">Par</th>
    <th data-key="tipo_red">Red</th>
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

let cur=[...ALL], sortKey="dist_km", sortAsc=true;

document.querySelectorAll("thead th[data-key]").forEach(th=>{{
  th.addEventListener("click",()=>{{
    const k=th.dataset.key;
    if(sortKey===k) sortAsc=!sortAsc; else{{sortKey=k;sortAsc=true;}}
    document.querySelectorAll("thead th").forEach(t=>t.classList.remove("asc","desc"));
    th.classList.add(sortAsc?"asc":"desc");
    render();
  }});
}});

function filter(){{
  const q  = document.getElementById("search").value.toLowerCase();
  const cf = document.getElementById("cat-filter").value;
  const tf = document.getElementById("tipo-filter").value;
  cur = ALL.filter(r=>{{
    const mq = !q || [r.from_nombre,r.to_nombre,r.from_cod,r.to_cod].some(v=>v.toLowerCase().includes(q));
    const mc = !cf || r.cat_par===cf;
    const mt = !tf || r.tipo_red===tf;
    return mq&&mc&&mt;
  }});
  render();
}}

function render(){{
  cur.sort((a,b)=>{{
    let av=a[sortKey],bv=b[sortKey];
    if(typeof av==="number") return sortAsc?av-bv:bv-av;
    return sortAsc?String(av).localeCompare(String(bv)):String(bv).localeCompare(String(av));
  }});
  const tb=document.getElementById("tbody");
  const f=document.createDocumentFragment();
  const MAX=3000;
  for(const r of cur.slice(0,MAX)){{
    const tr=document.createElement("tr");
    const tc=TC[r.tipo_red]||"";
    const pc_a=r.from_cat==="P"?"cat-P":"cat-S";
    const pc_b=r.to_cat  ==="P"?"cat-P":"cat-S";
    const ppc = r.cat_par==="PP"?"pp":r.cat_par==="SS"?"ss":"ps";
    tr.innerHTML=`
      <td><span class="nombre">${{r.from_nombre}}</span> <span class="cod">${{r.from_cod}}</span></td>
      <td><span class="${{pc_a}}">${{r.from_cat}}</span> <span class="tipo_dep">${{r.from_tipo_dep}}</span></td>
      <td><span class="nombre">${{r.to_nombre}}</span> <span class="cod">${{r.to_cod}}</span></td>
      <td><span class="${{pc_b}}">${{r.to_cat}}</span> <span class="tipo_dep">${{r.to_tipo_dep}}</span></td>
      <td class="dist">${{r.dist_km.toFixed(1)}}</td>
      <td class="${{ppc}}">${{r.cat_par}}</td>
      <td class="tipo ${{tc}}">${{r.tipo_red}}</td>
      <td class="linea" title="${{r.linea}}">${{r.linea}}</td>`;
    f.appendChild(tr);
  }}
  tb.innerHTML=""; tb.appendChild(f);
  document.getElementById("count").textContent =
    cur.length===ALL.length
      ? `${{ALL.length.toLocaleString()}} conexiones`
      : `${{cur.length.toLocaleString()}} de ${{ALL.length.toLocaleString()}} · mostrando ${{Math.min(MAX,cur.length).toLocaleString()}}`;
}}

document.querySelector("th[data-key='dist_km']").classList.add("asc");
render();
</script>
</body>
</html>
"""

html_path = os.path.join(BASE, "matriz_extendida.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = round(len(html)/1024)
print(f"HTML: {html_path} ({size_kb} KB)")
print(f"\nTop 10 conexiones más cortas:")
for r in results[:10]:
    flag = f"[{r['cat_par']}]"
    print(f"  {r['from_nombre']:30s} — {r['to_nombre']:30s}: {r['dist_km']:5.1f} km  {flag}")
