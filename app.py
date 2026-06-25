import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import subprocess
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

st.set_page_config(page_title="Pipeline Bancario", page_icon="🏦", layout="wide")

st.markdown("# 🏦 Pipeline Bancario — Predicción de Riesgo Crediticio")
st.markdown("Universidad del Pacífico · Grupo 2 · 2026-I")
st.divider()

# ── Constantes del pipeline ───────────────────
COLUMNA_ANOMALA   = "DAYS_EMPLOYED"
VALOR_ANOMALO     = 365243
UMBRAL_OBS        = 0.5
UMBRAL_ATTR       = 0.6

# ── Cargar modelo y metadata ─────────────────
@st.cache_resource
def cargar_modelo():
    with open("modelo_actual.pkl", "rb") as f:
        contenido = pickle.load(f)
    with open("modelo_metadata.json", "r") as f:
        metadata = json.load(f)
    return contenido, metadata

try:
    contenido_modelo, metadata = cargar_modelo()
    modelo   = contenido_modelo['modelo']
    tipo_mod = contenido_modelo['tipo']
except FileNotFoundError:
    st.error("No se encontró el modelo entrenado.")
    st.stop()

# ── Preprocesamiento para predicción ─────────
def preprocesar_para_prediccion(df_nuevo):
    """
    Preprocesa el CSV de predicción usando las columnas y categorías
    que el modelo tiene guardadas internamente (pandas_categorical).
    
    Enfoque:
    1. Corregir anomalía DAYS_EMPLOYED
    2. Limpiar nulos excesivos
    3. Imputar valores faltantes
    4. Alinear columnas exactamente a las del modelo
    5. Aplicar las pandas_categorical del modelo por índice
    
    NO depende del tamaño del train para detectar tipos de columnas —
    eso era el error raíz anterior (mismatch de 101 vs 98 categóricas).
    """
    df = df_nuevo.copy()
    ids = df['SK_ID_CURR'].values if 'SK_ID_CURR' in df.columns else np.arange(len(df))

    # Paso 1 — Corrección anomalía
    if COLUMNA_ANOMALA in df.columns:
        df[COLUMNA_ANOMALA] = df[COLUMNA_ANOMALA].replace(VALOR_ANOMALO, np.nan)

    # Quitar columnas no predictoras
    for col in ['TARGET', 'SK_ID_CURR']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Paso 2 — Limpiar filas con muchos nulos
    n_attr = df.shape[1]
    obs_vacios = (df.isnull().sum(axis=1) / n_attr) >= UMBRAL_OBS
    df = df.loc[~obs_vacios].copy()

    # Paso 3 — Alinear columnas al modelo ANTES de imputar
    cols_modelo = modelo.feature_name() if tipo_mod == 'lgb' else modelo.feature_names
    for col in cols_modelo:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols_modelo].copy()

    # Paso 4 — Imputar según tipo de columna
    # Las columnas categóricas las determinamos por las pandas_categorical del modelo
    # (no por detección automática que cambia según el tamaño del dataset)
    n_cats = len(modelo.pandas_categorical) if tipo_mod == 'lgb' else 0
    cols_cat_idx = list(range(n_cats))  # los primeros n_cats features son categóricos
    cols_cat  = [cols_modelo[i] for i in cols_cat_idx if i < len(cols_modelo)]
    cols_rest = [col for col in cols_modelo if col not in cols_cat]

    # Imputar categóricas con 'Desconocido'
    if cols_cat:
        imp_cat = SimpleImputer(strategy='constant', fill_value='Desconocido')
        df[cols_cat] = imp_cat.fit_transform(df[cols_cat].astype(str))

    # Imputar numéricas con mediana
    cols_num = [col for col in cols_rest if df[col].dtype in ['float64', 'int64', 'Int64', 'float32']]
    if cols_num:
        imp_num = SimpleImputer(strategy='median')
        df[cols_num] = imp_num.fit_transform(df[cols_num])

    # Paso 5 — Aplicar pandas_categorical del modelo por índice
    # Esto es lo que garantiza que las categorías coincidan exactamente
    for i, cats in enumerate(modelo.pandas_categorical):
        col = cols_modelo[i]
        df[col] = pd.Categorical(
            df[col].astype(str),
            categories=[str(c) for c in cats]
        )

    return df, ids


def predecir(df_listo):
    if tipo_mod == 'lgb':
        return modelo.predict(df_listo)
    else:
        dmat = xgb.DMatrix(df_listo, enable_categorical=True)
        return modelo.predict(dmat)


# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("## Modelo actual")
    st.success(f"**{metadata['config_id']}** ({metadata['modelo']})")
    st.markdown(f"**Versión:** `{metadata['version']}`")
    st.markdown(f"**AUC-ROC:** `{metadata['roc_auc']}`")
    st.markdown(f"**F1-Score:** `{metadata['f1_score']}`")
    st.divider()
    st.markdown("*El modelo se actualiza automáticamente via GitHub Actions cuando se sube nuevo dataset con el mensaje* `[entrenar]`")

# ── Tabs ─────────────────────────────────────
tab1, tab2 = st.tabs(["Predicción por lote (CSV)", "🔄 Reentrenar modelo"])

# ── TAB 1: Predicción ────────────────────────
with tab1:
    st.markdown("### Sube un CSV con múltiples clientes")
    st.markdown("""
    El archivo debe tener las mismas columnas que `application_test.csv` de Kaggle.
    No necesita columna `TARGET`. El modelo predice la probabilidad de incumplimiento.
    """)

    archivo = st.file_uploader("Subir CSV de predicción", type="csv", key="pred")

    if archivo:
        df_pred = pd.read_csv(archivo)
        st.markdown(f"**Registros cargados:** {len(df_pred):,}")
        st.dataframe(df_pred.head(5), use_container_width=True)

        if st.button("🚀 Predecir todos los registros", use_container_width=True):
            try:
                with st.spinner("Preprocesando y generando predicciones..."):
                    df_listo, ids = preprocesar_para_prediccion(df_pred)
                    probs = predecir(df_listo)

                n = min(len(ids), len(probs))
                resultado = pd.DataFrame({
                    'SK_ID_CURR': ids[:n],
                    'probabilidad_incumplimiento': np.round(probs[:n], 4),
                    'clasificacion': np.where(probs[:n] >= 0.5, 'ALTO RIESGO', 'BAJO RIESGO')
                })

                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total clientes", f"{len(resultado):,}")
                col2.metric("Alto riesgo", f"{(resultado['clasificacion']=='ALTO RIESGO').sum():,}")
                col3.metric("Bajo riesgo", f"{(resultado['clasificacion']=='BAJO RIESGO').sum():,}")

                st.dataframe(resultado.head(50), use_container_width=True)

                csv_out = resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Descargar resultados completos",
                    csv_out, "predicciones.csv", "text/csv"
                )

            except Exception as e:
                st.error(f"Error al predecir: {e}")

# ── TAB 2: Reentrenamiento ───────────────────
with tab2:
    st.markdown("### Reentrenar el modelo con nueva data")
    st.markdown("""
    Sube un CSV de entrenamiento (con columna `TARGET`).
    El sistema reemplazará `application_train.csv`, disparará el reentrenamiento
    automático via GitHub Actions y actualizará el modelo desplegado.
    """)
    st.warning("El CSV debe incluir la columna `TARGET` (0 = pagará, 1 = incumplirá).")

    archivo_train = st.file_uploader("Subir nuevo CSV de entrenamiento", type="csv", key="retrain")

    if archivo_train:
        df_new = pd.read_csv(archivo_train)
        st.markdown(f"**Registros cargados:** {len(df_new):,}")

        if 'TARGET' not in df_new.columns:
            st.error("El CSV no tiene columna TARGET.")
        else:
            conteo = df_new['TARGET'].value_counts()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total registros", f"{len(df_new):,}")
            col2.metric("Pagará (TARGET=0)", f"{int(conteo.get(0, conteo.get(0.0, 0))):,}")
            col3.metric("Incumplirá (TARGET=1)", f"{int(conteo.get(1, conteo.get(1.0, 0))):,}")

            st.dataframe(df_new.head(5), use_container_width=True)

            if st.button("🔄 Reemplazar datos y reentrenar", use_container_width=True):
                try:
                    with st.spinner("Guardando nuevo dataset..."):
                        df_new.to_csv("application_train.csv", index=False)

                    with st.spinner("Haciendo commit y push al repositorio..."):
                        token = os.getenv("PAT_TOKEN", "")
                        if not token:
                            st.error("No se encontró PAT_TOKEN. Revisa el docker-compose.yml.")
                            st.stop()

                        repo_url = f"https://x-access-token:{token}@github.com/SebastianGP1810/PipelineBancario_HPC_Cloud.git"

                        comandos = [
                            ["git", "config", "--global", "--add", "safe.directory", "/app"],
                            ["git", "config", "user.name", "Streamlit Bot"],
                            ["git", "config", "user.email", "streamlit@bot.com"],
                            ["git", "add", "application_train.csv"],
                            ["git", "commit", "-m", "actualizar dataset de entrenamiento [entrenar]"],
                            ["git", "push", repo_url, "HEAD:main"],
                        ]
                        for cmd in comandos:
                            r = subprocess.run(cmd, capture_output=True, text=True)
                            salida = r.stdout + r.stderr
                            if r.returncode != 0:
                                if cmd[1] == "commit" and "nothing to commit" in salida:
                                    continue
                                raise RuntimeError(f"'{' '.join(cmd[:2])}' falló: {salida.strip()}")

                    st.success("✅ Dataset subido. GitHub Actions está reentrenando el modelo.")
                    st.info("🕐 El proceso tarda ~35-40 min. Recarga la página al terminar para ver la nueva versión.")

                except Exception as e:
                    st.error(f"Error al hacer push al repositorio: {e}")