"""
App mejorada de Streamlit — Nivel de ríos/quebradas (CORNARE / MARCO)
--------------------------------------------------------------------  
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_BASE_URL = "https://marco.cornare.gov.co/api/v1/estaciones"

LLAVE_FECHA = "level_date"
LLAVE_VALOR = "level"
CANDIDATOS_LAT = ["lat", "latitude", "latitud"]
CANDIDATOS_LON = ["lng", "lon", "longitude", "longitud"]

st.set_page_config(page_title="Nivel de estación — Análisis Avanzado", page_icon="🌊", layout="wide")

# ------------------------------------------------------------------
# Funciones de consulta
# ------------------------------------------------------------------
def obtener_serie_nivel(codigo_estacion, desde, hasta, calidad=1, timeout=30):
    url = f"{API_BASE_URL}/{codigo_estacion}/nivel"
    params = {"desde": desde, "hasta": hasta, "calidad": calidad}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}"
    except requests.exceptions.RequestException as e:
        return None, f"Error de red: {e}"

def obtener_todas_las_paginas(datos_json, timeout=30):
    registros = list(datos_json.get("values", []))
    siguiente_url = datos_json.get("next")
    while siguiente_url:
        try:
            resp = requests.get(siguiente_url, timeout=timeout, verify=False)
        except requests.exceptions.RequestException:
            break
        if resp.status_code != 200:
            break
        pagina = resp.json()
        registros.extend(pagina.get("values", []))
        siguiente_url = pagina.get("next")
    return registros

def detectar_coordenadas(datos_json, lat_manual, lon_manual):
    """Busca lat/lon en la API, si falla usa las coordenadas ingresadas manualmente."""
    if not isinstance(datos_json, dict):
        return lat_manual, lon_manual, False

    lat = next((datos_json[k] for k in CANDIDATOS_LAT if k in datos_json), None)
    lon = next((datos_json[k] for k in CANDIDATOS_LON if k in datos_json), None)

    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon), True
        except (TypeError, ValueError):
            pass
    return lat_manual, lon_manual, False

def calcular_indice_calidad(df):
    if df.empty or len(df) < 2:
        return 0.0, 0, 0

    df_idx = df.set_index("fecha")
    frecuencia_tipica = df["fecha"].diff().dropna().mode()
    if len(frecuencia_tipica) == 0:
        return 0.0, 0, 0
    frecuencia_tipica = frecuencia_tipica[0]

    rango_completo = pd.date_range(start=df_idx.index.min(), end=df_idx.index.max(), freq=frecuencia_tipica)
    esperados = len(rango_completo)
    huecos = esperados - len(df_idx)
    completitud = max(0.0, 1 - (huecos / esperados)) if esperados > 0 else 0.0

    Q1, Q3 = df["nivel"].quantile(0.25), df["nivel"].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    es_outlier = (df["nivel"] < lim_inf) | (df["nivel"] > lim_sup) | (df["nivel"] < 0)
    proporcion_outliers = es_outlier.mean()

    indice = (completitud * 0.7 + (1 - proporcion_outliers) * 0.3) * 100
    return round(indice, 1), int(huecos), int(es_outlier.sum())

# ------------------------------------------------------------------
# Sidebar — Parámetros de personalización de tu estación
# ------------------------------------------------------------------
st.sidebar.header("🛠️ Configuración de Estación")
nombre_estudiante = st.sidebar.text_input("Investigador / Estudiante", "Tu Nombre")
nombre_estacion = st.sidebar.text_input("Nombre de la Estación", "Río Medellín - Punto A")
codigo_estacion = st.sidebar.text_input("Código de estación (CORNARE)", "42")

st.sidebar.subheader("Ubicación (Fallback si falla API)")
lat_manual = st.sidebar.number_input("Latitud", value=6.2766, format="%.6f")
lon_manual = st.sidebar.number_input("Longitud", value=-75.5901, format="%.6f")

st.sidebar.header("📅 Parámetros Temporales")
fecha_desde = st.sidebar.date_input("Desde", pd.to_datetime("2026-08-23")).strftime("%Y-%m-%d")
fecha_hasta = st.sidebar.date_input("Hasta", pd.to_datetime("2026-08-30")).strftime("%Y-%m-%d")
calidad = st.sidebar.selectbox("Calidad", [1, 0], index=0, help="1 = solo datos validados")
consultar = st.sidebar.button("🔍 Extraer Datos", type="primary")

st.title(f"🌊 Análisis de Nivel Hídrico: {nombre_estacion}")
st.caption(f"**Analista:** {nombre_estudiante} | **Código Sensor:** {codigo_estacion}")

# ------------------------------------------------------------------
# Consulta, Procesamiento y Transformación de Datos
# ------------------------------------------------------------------
if consultar:
    with st.spinner("Consultando la API y procesando datos..."):
        datos_crudos, error = obtener_serie_nivel(codigo_estacion, fecha_desde, fecha_hasta, calidad)

    if error:
        st.error(f"❌ {error}")
    else:
        registros = obtener_todas_las_paginas(datos_crudos)

        if not registros:
            st.warning("No hay registros para esta estación y rango de fechas.")
        else:
            # 1. Optimización de tipos de datos en Pandas
            df = pd.DataFrame(registros)
            df = df.rename(columns={LLAVE_FECHA: "fecha", LLAVE_VALOR: "nivel"})
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df["nivel"] = pd.to_numeric(df["nivel"], errors="coerce").astype(np.float32) # Optimización de memoria
            df = df.dropna(subset=["fecha", "nivel"]).sort_values("fecha").reset_index(drop=True)

            # 2. Análisis Estadístico (Z-Score y Min-Max)
            mean_nivel = df["nivel"].mean()
            std_nivel = df["nivel"].std()
            min_nivel = df["nivel"].min()
            max_nivel = df["nivel"].max()
            
            # Estandarización y Normalización
            df["z_score"] = ((df["nivel"] - mean_nivel) / std_nivel).astype(np.float32)
            df["min_max_norm"] = ((df["nivel"] - min_nivel) / (max_nivel - min_nivel)).astype(np.float32)

            lat, lon, coords_reales = detectar_coordenadas(datos_crudos, lat_manual, lon_manual)
            indice_calidad, huecos, n_outliers = calcular_indice_calidad(df)

            # --- Métricas principales (Dashboard) ---
            st.markdown("### 📊 Panel de Control y Calidad de Datos")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lecturas (N)", len(df))
            col2.metric("Nivel Promedio", f"{mean_nivel:.2f} m")
            col3.metric("Índice Calidad", f"{indice_calidad}%")
            col4.metric("Outliers (IQR)", n_outliers)

            # --- Vistas de Datos con Pestañas ---
            tab1, tab2, tab3 = st.tabs(["📈 Serie de Tiempo", "🔬 Análisis Estadístico", "📍 Ubicación Geográfica"])

            with tab1:
                st.subheader("Nivel Base del Agua")
                st.line_chart(df.set_index("fecha")["nivel"], color="#1f77b4")

            with tab2:
                st.subheader("Transformaciones de Datos")
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("**Estandarización Z-Score** (Media=0, Desviación=1)")
                    st.line_chart(df.set_index("fecha")["z_score"], color="#ff7f0e")
                
                with col_chart2:
                    st.markdown("**Normalización Min-Max** (Escala 0 a 1)")
                    st.line_chart(df.set_index("fecha")["min_max_norm"], color="#2ca02c")
                    
                st.caption("Estas transformaciones facilitan la comparación del comportamiento del caudal frente a otros fenómenos de diferentes escalas métricas.")

            with tab3:
                st.subheader(f"Geolocalización: {nombre_estacion}")
                if not coords_reales:
                    st.info("Utilizando coordenadas manuales definidas en la barra lateral.")
                st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=12)

            # --- Exportación y detalles ---
            st.divider()
            with st.expander("Ver y exportar conjunto de datos procesado"):
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar CSV Procesado", csv, file_name=f"{nombre_estacion.replace(' ', '_')}_datos.csv", mime="text/csv")
else:
    st.info("Ajusta los parámetros en el panel izquierdo y haz clic en **Extraer Datos**.")
