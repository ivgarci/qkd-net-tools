#!/usr/bin/env python3
"""
Genera red_adif_mapa.html — mapa interactivo red ferroviaria ADIF.
Nodos primarios : union de tres fuentes Renfe/ADIF, con conectividad verificada contra tramos
Nodos secundarios: adif_dependencias.geojson (puntos)
Trazados exactos : adif_tramos.geojson (LineStrings — geometría real)

Fuentes de estaciones:
  1. listado_completo_av_ld_md.csv  — Renfe AV/LD/MD
  2. listado-de-estaciones-feve-2.csv — FEVE (vía estrecha norte)
  3. estaciones.csv                 — catálogo completo ADIF/Renfe
"""

import json, csv, math, os
from datetime import date

BASE   = "/Users/igarcia/doctorado/2025_2026/mapas"
ADIF   = os.path.join(BASE, "adif")
RENFE  = os.path.join(BASE, "renfe")

# ─── Douglas-Peucker simplification (pure Python) ────────────────────────────

def point_line_dist(p, a, b):
    """Distancia perpendicular del punto p al segmento a-b (en grados²)."""
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx-ax, by-ay
    if dx == 0 and dy == 0:
        return math.hypot(px-ax, py-ay)
    t = ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t))
    return math.hypot(px - (ax+t*dx), py - (ay+t*dy))

def rdp(coords, eps):
    """Ramer-Douglas-Peucker."""
    if len(coords) < 3:
        return coords
    dmax, idx = 0, 0
    for i in range(1, len(coords)-1):
        d = point_line_dist(coords[i], coords[0], coords[-1])
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        l = rdp(coords[:idx+1], eps)
        r = rdp(coords[idx:],   eps)
        return l[:-1] + r
    return [coords[0], coords[-1]]

# ─── Cargar datos ─────────────────────────────────────────────────────────────

print("Cargando dependencias...")
with open(os.path.join(ADIF, "adif_dependencias.geojson"), encoding="utf-8") as f:
    dep_gj = json.load(f)

print("Cargando tramos...")
with open(os.path.join(ADIF, "adif_tramos.geojson"), encoding="utf-8") as f:
    tra_gj = json.load(f)

# ─── Conectividad contra tramos ──────────────────────────────────────────────
# Una estación está "conectada" si su código aparece como extremo de tramo
# o si su dependencia tiene un COD_TRAMO válido (estación intermedia).

print("Calculando conectividad de estaciones...")
tramo_endpoints = set()
tramo_by_codtramo = {f["properties"]["CODTRAMO"] for f in tra_gj["features"]
                     if f["properties"].get("CODTRAMO")}
for feat in tra_gj["features"]:
    p = feat["properties"]
    if p.get("DEPORIGEN"):  tramo_endpoints.add(p["DEPORIGEN"])
    if p.get("DEPDESTINO"): tramo_endpoints.add(p["DEPDESTINO"])

dep_linked = set()
for feat in dep_gj["features"]:
    p = feat["properties"]
    cod = p.get("COD_DEPEND","")
    ct  = p.get("COD_TRAMO","")
    if cod and ct and ct in tramo_by_codtramo:
        dep_linked.add(cod)

CONNECTED = tramo_endpoints | dep_linked
print(f"  Nodos conectados en red: {len(CONNECTED)}")

# ─── Cargar todas las fuentes de estaciones ──────────────────────────────────

BORDER_CODES = {"77310", "79316"}  # Latour-de-Carol, Cerbère

# Dict keyed by código; last-write wins for lower-priority sources
all_stations = {}   # codigo -> dict

def add_station(codigo, nombre, lat, lon, poblacion, provincia, pais, tipo):
    codigo = codigo.strip()
    if not codigo: return
    # Don't overwrite with lower-priority data if already present
    if codigo in all_stations:
        # Just update tipo flags (additive)
        all_stations[codigo]["tipo"].add(tipo)
        return
    all_stations[codigo] = {
        "codigo":    codigo,
        "nombre":    nombre.strip(),
        "lat":       lat,
        "lon":       lon,
        "poblacion": poblacion,
        "provincia": provincia,
        "pais":      pais,
        "tipo":      {tipo},   # set: may be {'av_ld_md'}, {'feve'}, etc.
    }

# 1. estaciones.csv — catálogo completo (lowest priority, loaded first)
print("Cargando estaciones.csv (catálogo completo)...")
n_est_spain = 0
with open(os.path.join(RENFE, "estaciones.csv"), encoding="latin-1") as f:
    for row in csv.DictReader(f, delimiter=";"):
        pais = row.get("PAIS","").strip().upper()
        if pais not in ("ESPAÑA", "ESPA\xd1A") and row.get("CODIGO","").strip().strip('"') not in BORDER_CODES:
            continue
        try:
            lat = float(row["LATITUD"].replace(",",".").strip('"'))
            lon = float(row["LONGITUD"].replace(",",".").strip('"'))
        except:
            continue
        feve_flag    = row.get("FEVE","").strip().upper() == "SI"
        cercan_flag  = row.get("CERCANIAS","").strip().upper() == "SI"
        tipo = "feve" if feve_flag else ("cercanias" if cercan_flag else "otra")
        add_station(
            row.get("CODIGO","").strip().strip('"'),
            row.get("DESCRIPCION","").strip().strip('"'),
            lat, lon,
            row.get("POBLACION","").strip().strip('"'),
            row.get("PROVINCIA","").strip().strip('"'),
            "España", tipo
        )
        n_est_spain += 1
print(f"  {n_est_spain} estaciones España en catálogo")

# 2. FEVE CSV — vía estrecha norte
print("Cargando FEVE...")
n_feve = 0
with open(os.path.join(RENFE, "listado-de-estaciones-feve-2.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f, delimiter=";"):
        code_key = [k for k in row.keys() if "DIGO" in k][0]
        codigo = row[code_key].strip()
        try:
            lat = float(row["LATITUD"].replace(",","."))
            lon = float(row["LONGITUD"].replace(",","."))
        except:
            continue
        add_station(codigo, row.get("DESCRIPCION",""),
                    lat, lon,
                    row.get("POBLACION",""), row.get("PROVINCIA",""),
                    "España", "feve")
        n_feve += 1
print(f"  {n_feve} estaciones FEVE")

# 3. Renfe AV/LD/MD — highest priority (overwrites tipo, keeps existing coord if better)
print("Cargando Renfe AV/LD/MD...")
n_avldmd = 0
dropped_foreign = 0
with open(os.path.join(RENFE, "listado_completo_av_ld_md.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f, delimiter=";"):
        code_key = [k for k in row.keys() if "CÓDIGO" in k or "DIGO" in k][0]
        codigo = row[code_key].strip()
        pais   = row.get("PAIS","").strip()
        if pais != "España" and codigo not in BORDER_CODES:
            dropped_foreign += 1
            continue
        try:
            lat = float(row["LATITUD"].replace(",","."))
            lon = float(row["LONGITUD"].replace(",","."))
        except:
            continue
        if codigo in all_stations:
            all_stations[codigo]["tipo"].add("av_ld_md")
            # Update coords with Renfe data (more precise for passenger stations)
            all_stations[codigo]["lat"] = lat
            all_stations[codigo]["lon"] = lon
        else:
            add_station(codigo, row.get("DESCRIPCION",""), lat, lon,
                        row.get("POBLACION",""), row.get("PROVINCIA",""),
                        pais, "av_ld_md")
        n_avldmd += 1
print(f"  {n_avldmd} estaciones AV/LD/MD (descartadas {dropped_foreign} extranjeras)")

# ─── Separar conectadas / desconectadas ──────────────────────────────────────

renfe_stations   = []   # connected → shown by default
disc_stations    = []   # disconnected → layer off, con aviso

for s in all_stations.values():
    tipo_set = s["tipo"]
    # Determine display color/label
    if "av_ld_md" in tipo_set:
        s["tipo_label"] = "AV/LD/MD"
        s["color"] = "#FFD700"
    elif "feve" in tipo_set:
        s["tipo_label"] = "FEVE"
        s["color"] = "#FF7733"
    elif "cercanias" in tipo_set:
        s["tipo_label"] = "Cercanías"
        s["color"] = "#66CC66"
    else:
        s["tipo_label"] = "Otra"
        s["color"] = "#44AACC"
    s["tipo"] = s["tipo_label"]  # convert set to string for JSON

    if s["codigo"] in CONNECTED:
        renfe_stations.append(s)
    else:
        disc_stations.append(s)

n_conn = len(renfe_stations)
n_disc = len(disc_stations)
print(f"  Conectadas: {n_conn}  |  Sin conexión en tramos: {n_disc}")

# ─── Dependencias (nodos secundarios) ────────────────────────────────────────

TIPO_D_LABEL = {
    "E":"Estación","P":"Punto KM","B":"Bifurcación","F":"Fin de línea",
    "T":"Túnel","Y":"Apartadero","A":"Apeadero","K":"Cargadero",
    "G":"Garita","O":"Otro","Z":"Zona técnica","C":"Cruce","D":"Depósito",
}

dep_nodes      = []   # en algún tramo
dep_disc_nodes = []   # sin tramo asociado
for feat in dep_gj["features"]:
    p = feat["properties"]; g = feat["geometry"]
    if not g: continue
    cod = p.get("COD_DEPEND","")
    node = {
        "cod":     cod,
        "nombre":  p.get("NOMBRE",""),
        "tipo":    TIPO_D_LABEL.get(p.get("COD_TIPO_D",""),""),
        "tipo_cod":p.get("COD_TIPO_D",""),
        "estado":  p.get("COD_ESTADO",""),
        "titular": p.get("COD_TITULA",""),
        "lat":     g["coordinates"][1],
        "lon":     g["coordinates"][0],
    }
    if cod in CONNECTED:
        dep_nodes.append(node)
    else:
        dep_disc_nodes.append(node)

print(f"  {len(dep_nodes)} dependencias conectadas | {len(dep_disc_nodes)} sin tramo")

# ─── Tramos con geometría simplificada ───────────────────────────────────────
# Tolerancia RDP: 0.003° ≈ 300m — suficiente para zoom 6-12

TIPO_RED_COLOR = {
    "Uso principal Alta Vel.":     "#FFD700",   # gold
    "Uso viaj. Interciud V>160":   "#FFA040",   # orange
    "Uso resto viaj. Interciud":   "#4499FF",   # blue
    "N.Cercanías >= 80 cir/día":   "#44DD88",   # green
    "N.Cercanías < 80 cir/día":    "#88DDAA",   # light green
    "Uso mercancías":              "#CC8844",   # brown-orange (VISIBLE)
    "Resto":                       "#AA6633",   # warm brown (VISIBLE)
}

TIPO_RED_WEIGHT = {
    "Uso principal Alta Vel.":     4.5,
    "Uso viaj. Interciud V>160":   3.5,
    "Uso resto viaj. Interciud":   2.5,
    "N.Cercanías >= 80 cir/día":   2,
    "N.Cercanías < 80 cir/día":    1.8,
    "Uso mercancías":              2,
    "Resto":                       1.8,
}

EPS = 0.003  # tolerancia RDP en grados

tramos_data = []
total_orig  = 0
total_simp  = 0

for feat in tra_gj["features"]:
    p = feat["properties"]; g = feat["geometry"]
    if not g or not g.get("coordinates"): continue
    coords_orig = g["coordinates"]
    total_orig += len(coords_orig)
    coords_simp = rdp(coords_orig, EPS)
    total_simp  += len(coords_simp)
    tramos_data.append({
        "id":       p.get("CODTRAMO",""),
        "linea":    p.get("COD_LINEA",""),
        "tipo_red": p.get("TIPO_RED",""),
        "orig":     p.get("DEPORIGEN",""),
        "dest":     p.get("DEPDESTINO",""),
        "longitud": p.get("LONGITUD",0),
        "coords":   coords_simp,
        "color":    TIPO_RED_COLOR.get(p.get("TIPO_RED",""), "#666666"),
        "weight":   TIPO_RED_WEIGHT.get(p.get("TIPO_RED",""), 1.8),
    })

reduction = 100*(1 - total_simp/total_orig) if total_orig else 0
print(f"  {len(tramos_data)} tramos | puntos: {total_orig:,} → {total_simp:,} ({reduction:.0f}% reducción, eps={EPS}°)")

# ─── Serializar a JS ─────────────────────────────────────────────────────────

renfe_js    = json.dumps(renfe_stations,  ensure_ascii=False, separators=(',',':'))
disc_js     = json.dumps(disc_stations,   ensure_ascii=False, separators=(',',':'))
dep_js      = json.dumps(dep_nodes,       ensure_ascii=False, separators=(',',':'))
dep_disc_js = json.dumps(dep_disc_nodes,  ensure_ascii=False, separators=(',',':'))
tramos_js   = json.dumps(tramos_data,     ensure_ascii=False, separators=(',',':'))
N_RENFE     = len(renfe_stations)
N_DISC      = len(disc_stations)
N_DEP       = len(dep_nodes)
N_DEP_DISC  = len(dep_disc_nodes)
N_TRAMOS    = len(tramos_data)
FECHA       = str(date.today())

# ─── HTML ─────────────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Red Ferroviaria ADIF – Fibra Oscura</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#eee;font-family:'Segoe UI',sans-serif}}
#map{{width:100vw;height:100vh}}
#info{{position:absolute;top:10px;left:10px;z-index:1000;background:rgba(0,0,0,.9);
       padding:12px 16px;border-radius:8px;max-width:260px;font-size:12px;border:1px solid #333}}
#info h3{{color:#FFD700;margin-bottom:8px;font-size:13px;letter-spacing:.5px}}
.stat{{color:#888;margin-bottom:3px}} .stat span{{color:#eee;font-weight:bold}}
hr{{border:none;border-top:1px solid #333;margin:8px 0}}
#panel{{position:absolute;bottom:20px;left:10px;z-index:1000;background:rgba(0,0,0,.9);
        padding:12px 14px;border-radius:8px;max-width:380px;font-size:11px;
        display:none;border:1px solid #444;max-height:60vh;overflow-y:auto}}
#panel h4{{color:#FFD700;margin-bottom:5px;font-size:12px}}
#panel p{{color:#ccc;margin:3px 0;line-height:1.5}}
#panel .lbl{{color:#888;font-size:10px}}
.close-btn{{float:right;cursor:pointer;color:#666;font-size:14px;line-height:1}}
.close-btn:hover{{color:#fff}}
.legend{{position:absolute;bottom:20px;right:10px;z-index:1000;background:rgba(0,0,0,.88);
          padding:10px 14px;border-radius:8px;font-size:11px;min-width:210px}}
.legend h4{{color:#FFD700;margin-bottom:6px;font-size:11px;letter-spacing:.3px}}
.leg-row{{display:flex;align-items:center;margin:4px 0}}
.leg-line{{height:3px;width:22px;margin-right:8px;border-radius:2px;flex-shrink:0}}
.leg-dot{{width:9px;height:9px;border-radius:50%;margin-right:8px;flex-shrink:0}}
.leg-note{{color:#777;font-size:10px;margin-top:4px;line-height:1.4}}
</style>
</head>
<body>
<div id="map"></div>

<div id="info">
  <h3>Red Ferroviaria ADIF · Fibra Oscura</h3>
  <div class="stat">Estaciones conectadas <span>{N_RENFE}</span></div>
  <div class="stat">Sin conexión en tramos <span style="color:#888">{N_DISC}</span></div>
  <div class="stat">Dependencias ADIF conectadas <span>{N_DEP}</span></div>
  <div class="stat">Dependencias sin tramo <span style="color:#888">{N_DEP_DISC}</span></div>
  <div class="stat">Tramos con trazado real <span>{N_TRAMOS}</span></div>
  <hr>
  <div class="stat">Fuente <span>ADIF FeatureService · Renfe 2026</span></div>
  <div class="stat">Fecha <span>{FECHA}</span></div>
  <hr>
  <div style="color:#555;font-size:10px">Clic en cualquier elemento para detalles</div>
</div>

<div id="panel">
  <span class="close-btn" onclick="this.parentNode.style.display='none'">✕</span>
  <div id="panel-body"></div>
</div>

<div class="legend">
  <h4>TIPO DE RED</h4>
  <div class="leg-row"><div class="leg-line" style="background:#FFD700;height:4.5px"></div>Alta Velocidad</div>
  <div class="leg-row"><div class="leg-line" style="background:#FFA040;height:3.5px"></div>Interciudad V&gt;160</div>
  <div class="leg-row"><div class="leg-line" style="background:#4499FF;height:2.5px"></div>Interciudad resto</div>
  <div class="leg-row"><div class="leg-line" style="background:#44DD88;height:2px"></div>Cercanías &ge;80</div>
  <div class="leg-row"><div class="leg-line" style="background:#88DDAA;height:1.8px"></div>Cercanías &lt;80</div>
  <div class="leg-row"><div class="leg-line" style="background:#CC8844;height:2px"></div>Mercancías</div>
  <div class="leg-row"><div class="leg-line" style="background:#AA6633;height:1.8px"></div>Resto</div>
  <div class="leg-note">Cercanías/Resto incluye antiguas líneas<br>FEVE (vía estrecha norte: Ferrol-Pravia,<br>Santander-Oviedo, etc.)</div>
  <hr>
  <div class="leg-row"><div class="leg-dot" style="background:#FFD700"></div>AV / LD / MD</div>
  <div class="leg-row"><div class="leg-dot" style="background:#FF7733"></div>FEVE (vía estrecha)</div>
  <div class="leg-row"><div class="leg-dot" style="background:#66CC66"></div>Cercanías</div>
  <div class="leg-row"><div class="leg-dot" style="background:#44AACC"></div>Otras estaciones</div>
  <div class="leg-row"><div class="leg-dot" style="background:#555;opacity:.7"></div>Sin datos de tramo</div>
  <div class="leg-row"><div class="leg-dot" style="background:#aaa;width:6px;height:6px"></div>Dependencia ADIF</div>
</div>

<script>
const RENFE     = {renfe_js};
const DISC      = {disc_js};
const DEPS      = {dep_js};
const DEPS_DISC = {dep_disc_js};
const TRAMOS    = {tramos_js};

const map = L.map('map').setView([40.2, -3.5], 6);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
  attribution:'&copy; CartoDB &copy; OpenStreetMap', maxZoom:19
}}).addTo(map);

const pb = document.getElementById('panel-body');
const panel = document.getElementById('panel');
function show(h) {{ pb.innerHTML = h; panel.style.display = 'block'; }}

// ── Capas ──────────────────────────────────────────────────────────────────
const layerTramos    = L.layerGroup().addTo(map);
const layerRenfe     = L.layerGroup().addTo(map);
const layerDisc      = L.layerGroup();   // estaciones sin tramo — off
const layerDeps      = L.layerGroup();   // dependencias conectadas — off
const layerDepsDisc  = L.layerGroup();   // dependencias sin tramo — off

// ── Tramos (geometría exacta) ─────────────────────────────────────────────
TRAMOS.forEach(t => {{
  const latlngs = t.coords.map(c => [c[1], c[0]]);
  L.polyline(latlngs, {{color:t.color, weight:t.weight, opacity:0.85}})
    .on('click', e => {{
      L.DomEvent.stopPropagation(e);
      const km = t.longitud ? (t.longitud/1000).toFixed(1)+' km' : '–';
      show(`<h4>${{t.linea}}</h4>
        <p><span class="lbl">Tipo red:</span> ${{t.tipo_red}}</p>
        <p><span class="lbl">Código tramo:</span> ${{t.id}}</p>
        <p><span class="lbl">Longitud:</span> ${{km}}</p>
        <p><span class="lbl">Origen → Destino:</span> ${{t.orig}} → ${{t.dest}}</p>`);
    }}).addTo(layerTramos);
}});

// ── Dependencias (nodos secundarios) ─────────────────────────────────────
DEPS.forEach(d => {{
  if(!d.lat || !d.lon) return;
  const col = d.estado === 'EX' ? '#aaaaaa' : '#444444';
  const r   = d.tipo_cod === 'E' ? 4 : d.tipo_cod === 'B' ? 3 : 2;
  L.circleMarker([d.lat, d.lon], {{radius:r, color:'#000', weight:0.5, fillColor:col, fillOpacity:0.75}})
    .on('click', e => {{
      L.DomEvent.stopPropagation(e);
      show(`<h4>${{d.nombre}}</h4>
        <p><span class="lbl">Código:</span> ${{d.cod}}</p>
        <p><span class="lbl">Tipo:</span> ${{d.tipo || d.tipo_cod}}</p>
        <p><span class="lbl">Titular:</span> ${{d.titular==='AV'?'Adif Alta Velocidad':d.titular==='AD'?'Adif convencional':d.titular}}</p>
        <p><span class="lbl">Estado:</span> ${{d.estado==='EX'?'En servicio':d.estado==='FS'?'Fuera de servicio':d.estado}}</p>
        <p style="color:#666;font-size:10px;margin-top:6px">Nodo secundario</p>`);
    }}).bindTooltip(d.nombre, {{permanent:false, direction:'top'}})
    .addTo(layerDeps);
}});

// ── Dependencias sin tramo ───────────────────────────────────────────────
DEPS_DISC.forEach(d => {{
  if(!d.lat || !d.lon) return;
  L.circleMarker([d.lat, d.lon], {{radius:2, color:'#222', weight:0.5, fillColor:'#444', fillOpacity:0.5}})
    .on('click', e => {{
      L.DomEvent.stopPropagation(e);
      show(`<h4 style="color:#aaa">${{d.nombre}}</h4>
        <p><span class="lbl">Código:</span> ${{d.cod}}</p>
        <p><span class="lbl">Tipo:</span> ${{d.tipo || d.tipo_cod}}</p>
        <p><span class="lbl">Estado:</span> ${{d.estado==='EX'?'En servicio':d.estado==='FS'?'Fuera de servicio':d.estado}}</p>
        <p style="color:#FF6644;font-size:10px;margin-top:6px">⚠ Dependencia sin tramo asociado</p>`);
    }}).bindTooltip(d.nombre+' ⚠', {{permanent:false, direction:'top'}})
    .addTo(layerDepsDisc);
}});

// ── Estaciones conectadas (nodos primarios) ──────────────────────────────
RENFE.forEach(s => {{
  if(!s.lat || !s.lon) return;
  L.circleMarker([s.lat, s.lon], {{
    radius:6, color:'#000', weight:1, fillColor:s.color, fillOpacity:0.95
  }}).on('click', e => {{
    L.DomEvent.stopPropagation(e);
    show(`<h4>${{s.nombre}}</h4>
      <p><span class="lbl">Código ADIF:</span> ${{s.codigo}}</p>
      <p><span class="lbl">Tipo:</span> ${{s.tipo}}</p>
      <p><span class="lbl">Municipio:</span> ${{s.poblacion}}</p>
      <p><span class="lbl">Provincia:</span> ${{s.provincia}}</p>
      <p style="color:${{s.color}};font-size:10px;margin-top:6px">Nodo primario — conectado en red</p>`);
  }}).bindTooltip(s.nombre, {{permanent:false, direction:'top'}})
    .addTo(layerRenfe);
}});

// ── Estaciones sin conexión en tramos ────────────────────────────────────
DISC.forEach(s => {{
  if(!s.lat || !s.lon) return;
  L.circleMarker([s.lat, s.lon], {{
    radius:4, color:'#333', weight:1, fillColor:'#555', fillOpacity:0.6
  }}).on('click', e => {{
    L.DomEvent.stopPropagation(e);
    show(`<h4 style="color:#aaa">${{s.nombre}}</h4>
      <p><span class="lbl">Código ADIF:</span> ${{s.codigo}}</p>
      <p><span class="lbl">Tipo:</span> ${{s.tipo}}</p>
      <p><span class="lbl">Provincia:</span> ${{s.provincia}}</p>
      <p style="color:#FF6644;font-size:10px;margin-top:6px">⚠ No aparece en datos de tramos ADIF</p>`);
  }}).bindTooltip(s.nombre+' ⚠', {{permanent:false, direction:'top'}})
    .addTo(layerDisc);
}});

// ── Control de capas ──────────────────────────────────────────────────────
L.control.layers(null, {{
  'Trazados ferroviarios':              layerTramos,
  'Estaciones conectadas':              layerRenfe,
  'Estaciones sin datos de tramo ⚠':   layerDisc,
  'Dependencias ADIF':                  layerDeps,
  'Dependencias sin tramo ⚠':          layerDepsDisc,
}}, {{collapsed:false}}).addTo(map);

map.on('click', () => {{ panel.style.display='none'; }});
</script>
</body>
</html>"""

out = os.path.join(BASE, "red_adif_mapa.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(out) // 1024
print(f"\n✓ red_adif_mapa.html ({size_kb} KB)")
print(f"  Estaciones conectadas    : {N_RENFE}")
print(f"  Estaciones sin tramo (⚠): {N_DISC}")
print(f"  Dependencias ADIF        : {N_DEP}")
print(f"  Tramos                   : {N_TRAMOS}")

# Show which stations are disconnected
if disc_stations:
    by_tipo = {}
    for s in disc_stations:
        by_tipo.setdefault(s["tipo"], []).append(s["nombre"])
    print("\nEstaciones sin conexión en tramos, por tipo:")
    for tipo, nombres in sorted(by_tipo.items()):
        print(f"  {tipo} ({len(nombres)}): {', '.join(nombres[:5])}{'...' if len(nombres)>5 else ''}")
