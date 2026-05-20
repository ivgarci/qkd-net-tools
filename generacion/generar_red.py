#!/usr/bin/env python3
"""
Red de Fibra Oscura — Península Ibérica
Fuentes: ENTSO-E Grid Map 2024 + ADIF/Reintel
Generado: 2026-05-17
"""

import json, csv, os, xml.etree.ElementTree as ET
from xml.dom import minidom

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, '..', 'datos', 'generado')
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# NODOS
# ─────────────────────────────────────────────
NODOS = [
    {"id":"N01","nombre":"Madrid Hub","lat":40.41,"lon":-3.70,"pais":"ES","region":"Madrid","tipo_red":"ambas","voltaje_kv":400,"tipo_linea":"AVE+Convencional","nivel_hub":1,"descripcion":"Nodo radial absoluto. Hub eléctrico 400 kV y ferroviario AVE. Sede Reintel."},
    {"id":"N02","nombre":"Zaragoza","lat":41.65,"lon":-0.90,"pais":"ES","region":"Aragón","tipo_red":"ambas","voltaje_kv":400,"tipo_linea":"AVE+Convencional","nivel_hub":2,"descripcion":"Cruce AVE Madrid-Barcelona y eje eléctrico NE-SW. 4 corredores convergentes."},
    {"id":"N03","nombre":"Barcelona","lat":41.38,"lon":2.14,"pais":"ES","region":"Cataluña","tipo_red":"ambas","voltaje_kv":400,"tipo_linea":"AVE+Convencional","nivel_hub":2,"descripcion":"Hub NE. Interconexión eléctrica HVDC con Francia (Baixas). Nodo AVE Madrid-BCN."},
    {"id":"N04","nombre":"Bilbao","lat":43.26,"lon":-2.93,"pais":"ES","region":"País Vasco","tipo_red":"ambas","voltaje_kv":400,"tipo_linea":"Convencional","nivel_hub":3,"descripcion":"Nodo eléctrico norte 400 kV. Interconexión Francia (Argia). Hub ferroviario cantábrico."},
    {"id":"N05","nombre":"Pamplona","lat":42.82,"lon":-1.64,"pais":"ES","region":"Navarra","tipo_red":"ambas","voltaje_kv":400,"tipo_linea":"Convencional","nivel_hub":3,"descripcion":"Corredor pirenaico central. Enlace Argia-Orcoyen 400 kV."},
    {"id":"N06","nombre":"Sevilla","lat":37.39,"lon":-5.97,"pais":"ES","region":"Andalucía","tipo_red":"ambas","voltaje_kv":400,"tipo_linea":"AVE+Convencional","nivel_hub":2,"descripcion":"Hub sur de España. AVE 1992. Confluyen Extremadura, Andalucía occidental y AVE Madrid."},
    {"id":"N07","nombre":"Valencia","lat":39.47,"lon":-0.38,"pais":"ES","region":"C. Valenciana","tipo_red":"ambas","voltaje_kv":400,"tipo_linea":"AVE+Convencional","nivel_hub":2,"descripcion":"Hub levantino. Corredor mediterráneo + AVE Madrid-Valencia."},
    {"id":"N08","nombre":"Lisboa","lat":38.72,"lon":-9.14,"pais":"PT","region":"Lisboa","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"Convencional","nivel_hub":2,"descripcion":"Hub central de Portugal. Subestación Carregado/Palmela 400 kV."},
    {"id":"N09","nombre":"Porto","lat":41.15,"lon":-8.61,"pais":"PT","region":"Norte PT","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"Convencional","nivel_hub":2,"descripcion":"Hub norte Portugal. Subestación Recarei/Valdigem. Corredor atlántico norte."},
    {"id":"N10","nombre":"Évora/Elvas (frontera PT-ES)","lat":38.57,"lon":-7.91,"pais":"PT","region":"Alentejo","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":3,"descripcion":"Interconexión principal Portugal-España. Doble circuito 400 kV (Évora-Almaraz/Cedillo)."},
    {"id":"N11","nombre":"Santiago de Compostela","lat":42.88,"lon":-8.54,"pais":"ES","region":"Galicia","tipo_red":"ambas","voltaje_kv":400,"tipo_linea":"AVE+Convencional","nivel_hub":3,"descripcion":"Hub gallego. Concentra eólica gallega. Bifurcación AVE Madrid-Galicia."},
    {"id":"N12","nombre":"Córdoba","lat":37.89,"lon":-4.78,"pais":"ES","region":"Andalucía","tipo_red":"ambas","voltaje_kv":400,"tipo_linea":"AVE","nivel_hub":2,"descripcion":"Nodo T crítico AVE Madrid-Sevilla/Málaga. Subestación eléctrica 400 kV Guillena-Mudéjar."},
    {"id":"N13","nombre":"Tarragona/Vandellós","lat":40.90,"lon":0.90,"pais":"ES","region":"Cataluña","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":3,"descripcion":"Hub costa mediterránea. Zona nuclear. Corredor 400 kV Barcelona-Valencia."},
    {"id":"N14","nombre":"Málaga","lat":36.71,"lon":-4.43,"pais":"ES","region":"Andalucía","tipo_red":"ambas","voltaje_kv":400,"tipo_linea":"AVE+Convencional","nivel_hub":3,"descripcion":"Terminus AVE Córdoba-Málaga. Corredor costero mediterráneo."},
    {"id":"N15","nombre":"Tarifa (cable submarino)","lat":36.01,"lon":-5.60,"pais":"ES","region":"Andalucía","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":3,"descripcion":"Único punto de interconexión ibérica con África. Cable submarino 400 kV al Estrecho."},
    {"id":"N16","nombre":"Valladolid","lat":41.64,"lon":-4.73,"pais":"ES","region":"Castilla y León","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE+Convencional","nivel_hub":3,"descripcion":"Nodo Norte-Galicia + Madrid. Hub Reintel Meseta Norte."},
    {"id":"N17","nombre":"Albacete","lat":38.99,"lon":-1.85,"pais":"ES","region":"Castilla-La Mancha","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE+Convencional","nivel_hub":3,"descripcion":"Bifurcación AVE Valencia/Alicante + ramal Murcia."},
    {"id":"N18","nombre":"Burgos","lat":42.34,"lon":-3.70,"pais":"ES","region":"Castilla y León","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":3,"descripcion":"Cruce Norte + País Vasco + Madrid."},
    {"id":"N19","nombre":"A Coruña","lat":43.37,"lon":-8.40,"pais":"ES","region":"Galicia","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE+Convencional","nivel_hub":3,"descripcion":"Terminus noroeste. Red gallega Reintel."},
    {"id":"N20","nombre":"Ourense","lat":42.34,"lon":-7.86,"pais":"ES","region":"Galicia","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE","nivel_hub":3,"descripcion":"Bifurcación AVE noroeste Santiago/A Coruña vs Vigo."},
    {"id":"N21","nombre":"León","lat":42.60,"lon":-5.57,"pais":"ES","region":"Castilla y León","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Cruce corredor Galicia-Asturias."},
    {"id":"N22","nombre":"Vitoria-Gasteiz","lat":42.85,"lon":-2.67,"pais":"ES","region":"País Vasco","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Nodo vasco. Corredor Y vasca futura."},
    {"id":"N23","nombre":"San Sebastián","lat":43.32,"lon":-1.98,"pais":"ES","region":"País Vasco","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Nodo fronterizo Francia ferroviario."},
    {"id":"N24","nombre":"Logroño","lat":42.47,"lon":-2.44,"pais":"ES","region":"La Rioja","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Eje del Ebro Zaragoza-Bilbao."},
    {"id":"N25","nombre":"Salamanca","lat":40.97,"lon":-5.66,"pais":"ES","region":"Castilla y León","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Cruce hacia Portugal vía Vilar Formoso."},
    {"id":"N26","nombre":"Cáceres","lat":39.47,"lon":-6.37,"pais":"ES","region":"Extremadura","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Corredor Madrid-Extremadura-Portugal."},
    {"id":"N27","nombre":"Badajoz","lat":38.88,"lon":-6.97,"pais":"ES","region":"Extremadura","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Frontera Portugal (Elvas). Nodo transfronterizo."},
    {"id":"N28","nombre":"Mérida","lat":38.92,"lon":-6.34,"pais":"ES","region":"Extremadura","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":5,"descripcion":"Interior extremeño Badajoz-Sevilla."},
    {"id":"N29","nombre":"Alicante","lat":38.34,"lon":-0.49,"pais":"ES","region":"C. Valenciana","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE","nivel_hub":3,"descripcion":"Terminus AVE corredor mediterráneo sur."},
    {"id":"N30","nombre":"Murcia","lat":37.98,"lon":-1.13,"pais":"ES","region":"Murcia","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Terminus sureste ferroviario."},
    {"id":"N31","nombre":"Almería","lat":36.84,"lon":-2.46,"pais":"ES","region":"Andalucía","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":5,"descripcion":"Terminus litoral sur. Dead-end Reintel."},
    {"id":"N32","nombre":"Granada","lat":37.18,"lon":-3.60,"pais":"ES","region":"Andalucía","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Interior Andalucía. Ramal Bobadilla-Granada."},
    {"id":"N33","nombre":"Huelva","lat":37.26,"lon":-6.95,"pais":"ES","region":"Andalucía","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":5,"descripcion":"Terminus occidental Andalucía."},
    {"id":"N34","nombre":"Cádiz","lat":36.52,"lon":-6.30,"pais":"ES","region":"Andalucía","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":5,"descripcion":"Terminus sur. Dead-end Reintel."},
    {"id":"N35","nombre":"Lleida","lat":41.62,"lon":0.62,"pais":"ES","region":"Cataluña","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE+Convencional","nivel_hub":4,"descripcion":"Corredor Madrid-Barcelona intermedio."},
    {"id":"N36","nombre":"Huesca","lat":42.14,"lon":-0.41,"pais":"ES","region":"Aragón","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":5,"descripcion":"Ramal pirenaico desde Zaragoza."},
    {"id":"N37","nombre":"Teruel","lat":40.34,"lon":-1.11,"pais":"ES","region":"Aragón","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":5,"descripcion":"Linea interior Zaragoza-Valencia."},
    {"id":"N38","nombre":"Jaén","lat":37.78,"lon":-3.79,"pais":"ES","region":"Andalucía","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":5,"descripcion":"Ramal interior desde corredor AVE."},
    {"id":"N39","nombre":"Gijón/Oviedo","lat":43.53,"lon":-5.66,"pais":"ES","region":"Asturias","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Terminus corredor cantábrico."},
    {"id":"N40","nombre":"Vigo","lat":42.23,"lon":-8.72,"pais":"ES","region":"Galicia","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Terminus sur Galicia. Conexión Portugal."},
    {"id":"N41","nombre":"Santa Llogaia","lat":42.30,"lon":2.87,"pais":"ES","region":"Cataluña","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":3,"descripcion":"Extremo español del HVDC Baixas-Santa Llogaia (interconexión ES-FR más importante)."},
    {"id":"N42","nombre":"Baixas (FR)","lat":42.62,"lon":2.83,"pais":"FR","region":"Occitanie","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":3,"descripcion":"Extremo francés interconexión HVDC con España."},
    {"id":"N43","nombre":"Argia (FR)","lat":43.35,"lon":-1.73,"pais":"FR","region":"Nouvelle-Aquitaine","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":3,"descripcion":"Extremo francés corredor pirenaico occidental."},
    {"id":"N44","nombre":"Orcoyen","lat":42.78,"lon":-1.67,"pais":"ES","region":"Navarra","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":3,"descripcion":"Extremo español corredor Argia-Orcoyen 400 kV."},
    {"id":"N45","nombre":"Valença (PT)","lat":41.93,"lon":-8.63,"pais":"PT","region":"Norte PT","tipo_red":"electrica","voltaje_kv":220,"tipo_linea":"-","nivel_hub":4,"descripcion":"Extremo portugués interconexión norte PT-ES 220 kV."},
    {"id":"N46","nombre":"Tuy","lat":42.05,"lon":-8.65,"pais":"ES","region":"Galicia","tipo_red":"electrica","voltaje_kv":220,"tipo_linea":"-","nivel_hub":4,"descripcion":"Extremo español interconexión norte ES-PT 220 kV."},
    {"id":"N47","nombre":"Tetuán (MA)","lat":35.57,"lon":-5.37,"pais":"MA","region":"Tánger-Tetuán","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":4,"descripcion":"Extremo marroquí cable submarino Estrecho de Gibraltar."},
    {"id":"N48","nombre":"Irún/Hendaya","lat":43.36,"lon":-1.79,"pais":"ES","region":"País Vasco","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Frontera ferroviaria norte con Francia."},
    {"id":"N49","nombre":"Portbou/Cervera","lat":42.43,"lon":3.16,"pais":"ES","region":"Cataluña","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional+AVE","nivel_hub":4,"descripcion":"Frontera ferroviaria mediterránea con Francia."},
    {"id":"N50","nombre":"Almussafes (subestación)","lat":39.30,"lon":-0.42,"pais":"ES","region":"C. Valenciana","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":3,"descripcion":"Subestación 400 kV Valencia-Almussafes. Hub eléctrico levantino."},
    {"id":"N51","nombre":"Almaraz/Cedillo","lat":39.66,"lon":-6.10,"pais":"ES","region":"Extremadura","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":3,"descripcion":"Subestación Cedillo 400 kV. Nodo eléctrico Extremadura. Interconexión PT-ES."},
    {"id":"N52","nombre":"Santander","lat":43.46,"lon":-3.80,"pais":"ES","region":"Cantabria","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Nodo corredor cantábrico."},
    {"id":"N53","nombre":"Lugo","lat":43.01,"lon":-7.55,"pais":"ES","region":"Galicia","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":5,"descripcion":"Nodo interior gallego A Coruña-Orense."},
    {"id":"N54","nombre":"Coimbra","lat":40.21,"lon":-8.43,"pais":"PT","region":"Centro PT","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Nodo intermedio corredor atlántico PT Porto-Lisboa."},
    {"id":"N55","nombre":"Faro","lat":37.02,"lon":-7.93,"pais":"PT","region":"Algarve","tipo_red":"electrica","voltaje_kv":220,"tipo_linea":"-","nivel_hub":5,"descripcion":"Terminus sur Portugal (Algarve)."},
    {"id":"N56","nombre":"Bobadilla","lat":37.06,"lon":-4.73,"pais":"ES","region":"Andalucía","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","nivel_hub":4,"descripcion":"Nodo ferroviario andaluz. Cruce Córdoba-Málaga-Granada-Sevilla."},
    {"id":"N57","nombre":"Bescanó","lat":41.95,"lon":2.84,"pais":"ES","region":"Cataluña","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":3,"descripcion":"Subestación 400 kV Girona. Hub eléctrico NE catalán."},
    {"id":"N58","nombre":"Galapagar","lat":40.57,"lon":-4.00,"pais":"ES","region":"Madrid","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":2,"descripcion":"Subestación 400 kV anillo NW Madrid. Parte del anillo perimetral madrileño."},
    {"id":"N59","nombre":"Moraleja de Enmedio","lat":40.27,"lon":-3.80,"pais":"ES","region":"Madrid","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":2,"descripcion":"Subestación 400 kV anillo SE Madrid."},
    {"id":"N60","nombre":"Guillena","lat":37.54,"lon":-6.05,"pais":"ES","region":"Andalucía","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","nivel_hub":3,"descripcion":"Subestación 400 kV hub Sevilla norte. Punto clave corredor Extremadura-Andalucía."},
]

# ─────────────────────────────────────────────
# CORREDORES (ARISTAS)
# ─────────────────────────────────────────────
CORREDORES = [
    # ELÉCTRICOS
    {"id":"E01","nombre":"Madrid – Bilbao 400 kV","origen":"N01","destino":"N04","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":450,"prob_fibra":"muy_alta","descripcion":"Corredor central Norte-Sur. Eje Madrid-Burgos-Vitoria-Bilbao."},
    {"id":"E02","nombre":"Barcelona – Valencia 400 kV","origen":"N03","destino":"N07","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":350,"prob_fibra":"muy_alta","descripcion":"Corredor mediterráneo norte. Pasa por Tarragona/Vandellós."},
    {"id":"E03","nombre":"Valencia – Málaga 400 kV","origen":"N07","destino":"N14","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":600,"prob_fibra":"alta","descripcion":"Corredor mediterráneo sur. Murcia-Almería-Málaga."},
    {"id":"E04","nombre":"HVDC Baixas – Santa Llogaia","origen":"N42","destino":"N41","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"HVDC","longitud_km":65,"prob_fibra":"muy_alta","descripcion":"Interconexión ES-FR más importante. Cable subterráneo HVDC por los Pirineos. Con fibra de señalización asociada."},
    {"id":"E05","nombre":"Argia – Orcoyen 400 kV","origen":"N43","destino":"N44","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":80,"prob_fibra":"muy_alta","descripcion":"Interconexión pirenaica occidental ES-FR."},
    {"id":"E06","nombre":"Madrid – Zaragoza 400 kV","origen":"N01","destino":"N02","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":300,"prob_fibra":"muy_alta","descripcion":"Corredor diagonal NE. Eje eléctrico Madrid-NE."},
    {"id":"E07","nombre":"Madrid – Sevilla 400 kV","origen":"N01","destino":"N06","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":500,"prob_fibra":"muy_alta","descripcion":"Corredor Sur. Pasa por Extremadura/Guillena."},
    {"id":"E08","nombre":"Porto – Lisboa 400 kV","origen":"N09","destino":"N08","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":300,"prob_fibra":"muy_alta","descripcion":"Corredor atlántico Portugal. Pasa por Coimbra."},
    {"id":"E09","nombre":"Madrid – Évora/Elvas 400 kV","origen":"N01","destino":"N10","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":550,"prob_fibra":"alta","descripcion":"Interconexión ES-PT vía Extremadura. Pasa por Cedillo/Almaraz."},
    {"id":"E10","nombre":"Galicia – Madrid 400 kV","origen":"N11","destino":"N01","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":500,"prob_fibra":"alta","descripcion":"Evacuación eólica gallega hacia Madrid. Corredor NW-SE."},
    {"id":"E11","nombre":"Tuy – Valença 220 kV","origen":"N46","destino":"N45","tipo_red":"electrica","voltaje_kv":220,"tipo_linea":"-","longitud_km":10,"prob_fibra":"alta","descripcion":"Interconexión norte PT-ES 220 kV sobre río Miño."},
    {"id":"E12","nombre":"Cable submarino Tarifa – Tetuán","origen":"N15","destino":"N47","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"submarino","longitud_km":28,"prob_fibra":"media","descripcion":"Única interconexión ibérica con África. Cable submarino 400 kV Estrecho de Gibraltar."},
    {"id":"E13","nombre":"Zaragoza – Barcelona 400 kV","origen":"N02","destino":"N03","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":300,"prob_fibra":"muy_alta","descripcion":"Corredor NE. Pasa por Lleida/Vandellós."},
    {"id":"E14","nombre":"Sevilla – Córdoba 400 kV","origen":"N06","destino":"N12","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":140,"prob_fibra":"muy_alta","descripcion":"Eje Andalucía occidental-central."},
    {"id":"E15","nombre":"Bilbao – Pamplona 220 kV","origen":"N04","destino":"N05","tipo_red":"electrica","voltaje_kv":220,"tipo_linea":"-","longitud_km":100,"prob_fibra":"alta","descripcion":"Corredor cantábrico este."},
    {"id":"E16","nombre":"Madrid – Pamplona 400 kV","origen":"N01","destino":"N05","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":330,"prob_fibra":"alta","descripcion":"Corredor Norte via Zaragoza-Navarra."},
    {"id":"E17","nombre":"Corredor Cantábrico Eléctrico","origen":"N04","destino":"N39","tipo_red":"electrica","voltaje_kv":220,"tipo_linea":"-","longitud_km":250,"prob_fibra":"media","descripcion":"Corredor 220 kV Bilbao-Santander-Asturias."},
    {"id":"E18","nombre":"Évora – Lisboa 400 kV","origen":"N10","destino":"N08","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":130,"prob_fibra":"muy_alta","descripcion":"Enlace Alentejo-Lisboa Portugal."},
    {"id":"E19","nombre":"Galapagar – Madrid anillo 400 kV","origen":"N58","destino":"N01","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":40,"prob_fibra":"muy_alta","descripcion":"Anillo perimetral Madrid NW 400 kV."},
    {"id":"E20","nombre":"Guillena – Sevilla 400 kV","origen":"N60","destino":"N06","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":25,"prob_fibra":"muy_alta","descripcion":"Subestación hub norte Sevilla."},
    {"id":"E21","nombre":"Bescanó – Barcelona 400 kV","origen":"N57","destino":"N03","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":80,"prob_fibra":"muy_alta","descripcion":"Corredor Girona-Barcelona 400 kV."},
    {"id":"E22","nombre":"Santa Llogaia – Bescanó 400 kV","origen":"N41","destino":"N57","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":50,"prob_fibra":"muy_alta","descripcion":"Enlace interno NE catalán."},
    {"id":"E23","nombre":"Cedillo – Madrid 400 kV","origen":"N51","destino":"N01","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":280,"prob_fibra":"alta","descripcion":"Corredor Extremadura-Madrid 400 kV."},
    {"id":"E24","nombre":"Porto – Galicia 400 kV","origen":"N09","destino":"N11","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":120,"prob_fibra":"alta","descripcion":"Interconexión eléctrica Galicia-Norte Portugal."},
    {"id":"E25","nombre":"Tarragona – Valencia 400 kV","origen":"N13","destino":"N07","tipo_red":"electrica","voltaje_kv":400,"tipo_linea":"-","longitud_km":200,"prob_fibra":"muy_alta","descripcion":"Corredor mediterráneo central. Pasa por costa de Castellón."},
    # FERROVIARIOS (REINTEL)
    {"id":"R01","nombre":"AVE Madrid – Barcelona","origen":"N01","destino":"N03","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE","longitud_km":650,"prob_fibra":"muy_alta","descripcion":"Corredor AVE más importante. Pasa por Zaragoza y Lleida."},
    {"id":"R02","nombre":"AVE Madrid – Sevilla","origen":"N01","destino":"N06","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE","longitud_km":471,"prob_fibra":"muy_alta","descripcion":"Primer AVE español (1992). Pasa por Córdoba."},
    {"id":"R03","nombre":"AVE Córdoba – Málaga","origen":"N12","destino":"N14","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE","longitud_km":155,"prob_fibra":"alta","descripcion":"Ramal AVE Málaga."},
    {"id":"R04","nombre":"AVE Madrid – Valencia","origen":"N01","destino":"N07","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE","longitud_km":391,"prob_fibra":"muy_alta","descripcion":"Corredor AVE Levante. Pasa por Albacete."},
    {"id":"R05","nombre":"AVE Madrid – Alicante","origen":"N01","destino":"N29","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE","longitud_km":438,"prob_fibra":"muy_alta","descripcion":"Corredor AVE sur Levante. Bifurca en Albacete desde R04."},
    {"id":"R06","nombre":"AVE Madrid – Santiago","origen":"N01","destino":"N11","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE_parcial","longitud_km":800,"prob_fibra":"alta","descripcion":"Corredor Galicia. Pasa por Valladolid, León y Ourense."},
    {"id":"R07","nombre":"Corredor Cantábrico Norte","origen":"N04","destino":"N48","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":100,"prob_fibra":"alta","descripcion":"Bilbao-San Sebastián-Irún. Frontera Francia."},
    {"id":"R08","nombre":"Eje del Ebro","origen":"N18","destino":"N02","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":420,"prob_fibra":"alta","descripcion":"Miranda-Logroño-Zaragoza-Lleida. Corredor fluvial del Ebro."},
    {"id":"R09","nombre":"Corredor Mediterráneo AVE","origen":"N03","destino":"N29","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE_parcial","longitud_km":650,"prob_fibra":"muy_alta","descripcion":"Barcelona-Tarragona-Valencia-Alicante. Parcialmente AVE."},
    {"id":"R10","nombre":"Madrid – Lisboa ferroviario","origen":"N01","destino":"N08","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":640,"prob_fibra":"alta","descripcion":"Vía Cáceres-Badajoz-Elvas. Corredor internacional Ibérico."},
    {"id":"R11","nombre":"Red gallega Vigo – A Coruña","origen":"N40","destino":"N19","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE_parcial","longitud_km":170,"prob_fibra":"alta","descripcion":"Corredor costero gallego. Pasa por Santiago."},
    {"id":"R12","nombre":"Madrid – Burgos – Norte","origen":"N01","destino":"N18","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":250,"prob_fibra":"alta","descripcion":"Corredor Norte. Pasa por Valladolid."},
    {"id":"R13","nombre":"Sevilla – Cádiz","origen":"N06","destino":"N34","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":130,"prob_fibra":"media","descripcion":"Ramal convencional sur Andalucía."},
    {"id":"R14","nombre":"Valencia – Murcia – Almería","origen":"N07","destino":"N31","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":400,"prob_fibra":"media","descripcion":"Corredor sureste ferroviario convencional."},
    {"id":"R15","nombre":"Madrid – Valladolid AVE","origen":"N01","destino":"N16","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE","longitud_km":200,"prob_fibra":"alta","descripcion":"Primer tramo AVE Madrid-Galicia."},
    {"id":"R16","nombre":"Frontera ferroviaria mediterránea","origen":"N03","destino":"N49","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE+Convencional","longitud_km":150,"prob_fibra":"alta","descripcion":"Barcelona-Portbou. Conexión ferroviaria con Francia por Mediterráneo."},
    {"id":"R17","nombre":"Irún – San Sebastián","origen":"N48","destino":"N23","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":20,"prob_fibra":"alta","descripcion":"Conexión frontera norte Francia con red española."},
    {"id":"R18","nombre":"Vigo – Valença ferroviario","origen":"N40","destino":"N45","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":40,"prob_fibra":"alta","descripcion":"Interconexión ferroviaria Galicia-Norte Portugal."},
    {"id":"R19","nombre":"Ourense – Vigo","origen":"N20","destino":"N40","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE","longitud_km":90,"prob_fibra":"alta","descripcion":"Ramal AVE sur gallego."},
    {"id":"R20","nombre":"Ourense – Santiago","origen":"N20","destino":"N11","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE","longitud_km":110,"prob_fibra":"alta","descripcion":"AVE Galicia tramo interior."},
    {"id":"R21","nombre":"León – Gijón","origen":"N21","destino":"N39","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":135,"prob_fibra":"media","descripcion":"Corredor Asturias-Meseta por puerto de Pajares."},
    {"id":"R22","nombre":"Valladolid – León","origen":"N16","destino":"N21","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":120,"prob_fibra":"alta","descripcion":"Tramo corredor Madrid-Galicia."},
    {"id":"R23","nombre":"Burgos – Vitoria","origen":"N18","destino":"N22","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":110,"prob_fibra":"alta","descripcion":"Corredor Y vasca futura. Actualmente convencional."},
    {"id":"R24","nombre":"Vitoria – San Sebastián","origen":"N22","destino":"N23","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":100,"prob_fibra":"alta","descripcion":"Corredor vasco costero."},
    {"id":"R25","nombre":"Zaragoza – Logroño","origen":"N02","destino":"N24","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":170,"prob_fibra":"media","descripcion":"Eje del Ebro medio."},
    {"id":"R26","nombre":"Logroño – Vitoria","origen":"N24","destino":"N22","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":100,"prob_fibra":"media","descripcion":"Enlace La Rioja-País Vasco."},
    {"id":"R27","nombre":"Zaragoza – Huesca","origen":"N02","destino":"N36","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":72,"prob_fibra":"media","descripcion":"Ramal pirenaico aragonés."},
    {"id":"R28","nombre":"Zaragoza – Teruel – Valencia","origen":"N02","destino":"N07","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":350,"prob_fibra":"media","descripcion":"Línea interior Aragón-Levante. Pasa por Teruel."},
    {"id":"R29","nombre":"Albacete – Murcia","origen":"N17","destino":"N30","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":130,"prob_fibra":"media","descripcion":"Ramal sureste desde bifurcación Albacete."},
    {"id":"R30","nombre":"Sevilla – Huelva","origen":"N06","destino":"N33","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":95,"prob_fibra":"baja","descripcion":"Ramal occidental Andalucía."},
    {"id":"R31","nombre":"Sevilla – Granada (vía Bobadilla)","origen":"N06","destino":"N32","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":260,"prob_fibra":"media","descripcion":"Vía Bobadilla. Nodo intermedio clave."},
    {"id":"R32","nombre":"Bobadilla – Almería","origen":"N56","destino":"N31","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":220,"prob_fibra":"media","descripcion":"Corredor litoral sur Andalucía."},
    {"id":"R33","nombre":"Madrid – Salamanca – Portugal","origen":"N01","destino":"N25","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":210,"prob_fibra":"media","descripcion":"Ramal occidental Madrid hacia Salamanca y frontera PT."},
    {"id":"R34","nombre":"Salamanca – Cáceres","origen":"N25","destino":"N26","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":180,"prob_fibra":"media","descripcion":"Corredor extremeño norte."},
    {"id":"R35","nombre":"Cáceres – Badajoz","origen":"N26","destino":"N27","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":90,"prob_fibra":"media","descripcion":"Corredor Extremadura sur hacia Portugal."},
    {"id":"R36","nombre":"Porto – Lisboa ferroviario","origen":"N09","destino":"N08","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":310,"prob_fibra":"alta","descripcion":"Corredor atlántico ferroviario Portugal. Pasa por Coimbra."},
    {"id":"R37","nombre":"A Coruña – Lugo","origen":"N19","destino":"N53","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":100,"prob_fibra":"media","descripcion":"Corredor interior gallego noroeste."},
    {"id":"R38","nombre":"Lugo – Ourense","origen":"N53","destino":"N20","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":100,"prob_fibra":"media","descripcion":"Corredor interior gallego sur."},
    {"id":"R39","nombre":"Santander – Bilbao","origen":"N52","destino":"N04","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":100,"prob_fibra":"media","descripcion":"Corredor cantábrico ferroviario."},
    {"id":"R40","nombre":"Santander – Burgos","origen":"N52","destino":"N18","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":155,"prob_fibra":"media","descripcion":"Enlace cantábrico-meseta por puerto del Escudo."},
    {"id":"R41","nombre":"Lleida – Barcelona","origen":"N35","destino":"N03","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE","longitud_km":160,"prob_fibra":"muy_alta","descripcion":"Tramo final AVE Madrid-Barcelona."},
    {"id":"R42","nombre":"Lleida – Zaragoza","origen":"N35","destino":"N02","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE","longitud_km":150,"prob_fibra":"muy_alta","descripcion":"Tramo central AVE Madrid-Barcelona."},
    {"id":"R43","nombre":"Murcia – Cartagena","origen":"N30","destino":"N30","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"Convencional","longitud_km":45,"prob_fibra":"baja","descripcion":"Ramal terminal Cartagena."},
    {"id":"R44","nombre":"Sevilla – Cáceres (AVE futura)","origen":"N06","destino":"N26","tipo_red":"ferroviaria","voltaje_kv":0,"tipo_linea":"AVE_futura","longitud_km":260,"prob_fibra":"alta","descripcion":"Corredor Extremadura AVE en proyecto. Ya con fibra en parte del trazado."},
]

# ─────────────────────────────────────────────
# GENERACIÓN DE ARCHIVOS
# ─────────────────────────────────────────────

def write_csv_nodos():
    path = os.path.join(OUT, "nodos.csv")
    fields = ["id","nombre","lat","lon","pais","region","tipo_red","voltaje_kv","tipo_linea","nivel_hub","descripcion"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(NODOS)
    print(f"✓ {path}")

def write_csv_corredores():
    path = os.path.join(OUT, "corredores.csv")
    fields = ["id","nombre","origen","destino","tipo_red","voltaje_kv","tipo_linea","longitud_km","prob_fibra","descripcion"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(CORREDORES)
    print(f"✓ {path}")

def write_geojson_nodos():
    nodo_map = {n["id"]: n for n in NODOS}
    features = []
    for n in NODOS:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [n["lon"], n["lat"]]},
            "properties": {k: v for k, v in n.items() if k not in ("lat","lon")}
        })
    fc = {"type": "FeatureCollection", "features": features}
    path = os.path.join(OUT, "nodos.geojson")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    print(f"✓ {path}")
    return nodo_map

def write_geojson_corredores(nodo_map):
    features = []
    for c in CORREDORES:
        o = nodo_map.get(c["origen"])
        d = nodo_map.get(c["destino"])
        if not o or not d:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [
                [o["lon"], o["lat"]], [d["lon"], d["lat"]]
            ]},
            "properties": {k: v for k, v in c.items() if k not in ("origen","destino")}
        })
    fc = {"type": "FeatureCollection", "features": features}
    path = os.path.join(OUT, "corredores.geojson")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    print(f"✓ {path}")

def write_geojson_combined(nodo_map):
    features = []
    for n in NODOS:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [n["lon"], n["lat"]]},
            "properties": {"feature_type": "nodo", **{k: v for k, v in n.items() if k not in ("lat","lon")}}
        })
    for c in CORREDORES:
        o = nodo_map.get(c["origen"])
        d = nodo_map.get(c["destino"])
        if not o or not d:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [
                [o["lon"], o["lat"]], [d["lon"], d["lat"]]
            ]},
            "properties": {"feature_type": "corredor", **{k: v for k, v in c.items() if k not in ("origen","destino")}}
        })
    fc = {
        "type": "FeatureCollection",
        "name": "Red Fibra Oscura Peninsula Iberica",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }
    path = os.path.join(OUT, "red_iberica.geojson")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    print(f"✓ {path}")

def write_kml(nodo_map):
    kml_header = '<?xml version="1.0" encoding="UTF-8"?>\n'
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = "Red Fibra Oscura – Península Ibérica"
    ET.SubElement(doc, "description").text = "Nodos y corredores de fibra oscura (proxy ENTSO-E + ADIF/Reintel). Generado 2026-05-17."

    # Estilos
    for style_id, color in [("electrica","ff0000ff"),("ferroviaria","ff00aa00"),("ambas","ff0088ff"),("frontera","ffff8800")]:
        st = ET.SubElement(doc, "Style", id=style_id)
        is_ = ET.SubElement(st, "IconStyle")
        ET.SubElement(is_, "color").text = color
        ET.SubElement(is_, "scale").text = "1.0"
        ico = ET.SubElement(is_, "Icon")
        ET.SubElement(ico, "href").text = "http://maps.google.com/mapfiles/kml/shapes/donut.png"
        ls = ET.SubElement(st, "LineStyle")
        ET.SubElement(ls, "color").text = color
        ET.SubElement(ls, "width").text = "2"

    # Carpeta nodos
    fn = ET.SubElement(doc, "Folder")
    ET.SubElement(fn, "name").text = "Nodos"
    for n in NODOS:
        pm = ET.SubElement(fn, "Placemark")
        ET.SubElement(pm, "name").text = n["nombre"]
        ET.SubElement(pm, "description").text = n["descripcion"]
        style_url = n["tipo_red"] if n["tipo_red"] in ("electrica","ferroviaria","ambas") else "ambas"
        if n["pais"] != "ES" and n["pais"] != "PT":
            style_url = "frontera"
        ET.SubElement(pm, "styleUrl").text = f"#{style_url}"
        pt = ET.SubElement(pm, "Point")
        ET.SubElement(pt, "coordinates").text = f"{n['lon']},{n['lat']},0"

    # Carpeta corredores
    fc = ET.SubElement(doc, "Folder")
    ET.SubElement(fc, "name").text = "Corredores"
    for c in CORREDORES:
        o = nodo_map.get(c["origen"])
        d = nodo_map.get(c["destino"])
        if not o or not d:
            continue
        pm = ET.SubElement(fc, "Placemark")
        ET.SubElement(pm, "name").text = c["nombre"]
        ET.SubElement(pm, "description").text = c["descripcion"]
        ET.SubElement(pm, "styleUrl").text = f"#{c['tipo_red']}" if c["tipo_red"] in ("electrica","ferroviaria") else "#ambas"
        ls = ET.SubElement(pm, "LineString")
        ET.SubElement(ls, "tessellate").text = "1"
        ET.SubElement(ls, "coordinates").text = f"{o['lon']},{o['lat']},0 {d['lon']},{d['lat']},0"

    raw = ET.tostring(kml, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    path = os.path.join(OUT, "red_iberica.kml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(pretty)
    print(f"✓ {path}")

def write_graphml(nodo_map):
    root = ET.Element("graphml", {
        "xmlns": "http://graphml.graphstruct.org/graphml",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://graphml.graphstruct.org/graphml http://graphml.graphstruct.org/graphml/1.0/graphml.xsd"
    })
    # Keys para nodos
    for key_id, attr_name, attr_type in [
        ("d0","nombre","string"),("d1","lat","double"),("d2","lon","double"),
        ("d3","pais","string"),("d4","region","string"),("d5","tipo_red","string"),
        ("d6","voltaje_kv","int"),("d7","nivel_hub","int"),("d8","descripcion","string"),
    ]:
        ET.SubElement(root, "key", id=key_id, **{"for":"node","attr.name":attr_name,"attr.type":attr_type})
    # Keys para aristas
    for key_id, attr_name, attr_type in [
        ("e0","nombre","string"),("e1","tipo_red","string"),("e2","voltaje_kv","int"),
        ("e3","tipo_linea","string"),("e4","longitud_km","int"),("e5","prob_fibra","string"),
        ("e6","descripcion","string"),
    ]:
        ET.SubElement(root, "key", id=key_id, **{"for":"edge","attr.name":attr_name,"attr.type":attr_type})

    graph = ET.SubElement(root, "graph", id="red_iberica", edgedefault="undirected")

    for n in NODOS:
        node_el = ET.SubElement(graph, "node", id=n["id"])
        for key_id, field in [("d0","nombre"),("d3","pais"),("d4","region"),
                               ("d5","tipo_red"),("d8","descripcion")]:
            d = ET.SubElement(node_el, "data", key=key_id)
            d.text = str(n[field])
        for key_id, field in [("d1","lat"),("d2","lon"),("d6","voltaje_kv"),("d7","nivel_hub")]:
            d = ET.SubElement(node_el, "data", key=key_id)
            d.text = str(n[field])

    edge_count = 0
    for c in CORREDORES:
        if c["origen"] == c["destino"]:
            continue
        edge_el = ET.SubElement(graph, "edge", id=c["id"], source=c["origen"], target=c["destino"])
        for key_id, field in [("e0","nombre"),("e1","tipo_red"),("e3","tipo_linea"),
                               ("e5","prob_fibra"),("e6","descripcion")]:
            d = ET.SubElement(edge_el, "data", key=key_id)
            d.text = str(c[field])
        for key_id, field in [("e2","voltaje_kv"),("e4","longitud_km")]:
            d = ET.SubElement(edge_el, "data", key=key_id)
            d.text = str(c[field])
        edge_count += 1

    raw = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    path = os.path.join(OUT, "red_iberica.graphml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(pretty)
    print(f"✓ {path}  ({len(NODOS)} nodos, {edge_count} aristas)")

def write_json():
    data = {
        "meta": {
            "titulo": "Red de Fibra Oscura – Península Ibérica",
            "fuentes": ["ENTSO-E Grid Map 2024", "ADIF/Reintel"],
            "fecha": "2026-05-17",
            "notas": "Coordenadas aproximadas. Los corredores son trazados rectos entre nodos; el trazado real sigue la infraestructura eléctrica/ferroviaria."
        },
        "nodos": NODOS,
        "corredores": CORREDORES
    }
    path = os.path.join(OUT, "red_iberica.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ {path}")

if __name__ == "__main__":
    print(f"\nGenerando archivos en: {OUT}\n")
    write_csv_nodos()
    write_csv_corredores()
    nodo_map = write_geojson_nodos()
    write_geojson_corredores(nodo_map)
    write_geojson_combined(nodo_map)
    write_kml(nodo_map)
    write_graphml(nodo_map)
    write_json()
    print(f"\nTotal: {len(NODOS)} nodos, {len(CORREDORES)} corredores generados.\n")
