import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import subprocess
import xgboost as xgb

# Reutilizamos EXACTAMENTE las mismas funciones de preprocesamiento
# que usa train_and_export.py para entrenar. Esto garantiza que los
# datos que llegan al modelo tengan el mismo formato que en el entrenamiento.
from train_and_export import (
    limpiar_nulos_excesivos,
    preprocesar_datos,
    COLUMNA_ANOMALA,
    VALOR_ANOMALO,
    UMBRAL_CATEGORICA,
)

st.set_page_config(page_title="Pipeline Bancario", page_icon="🏦", layout="wide")

st.markdown("# 🏦 Pipeline Bancario — Predicción de Riesgo Crediticio")
st.markdown("Universidad del Pacífico · Grupo 2 · 2026-I")
st.divider()

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
    st.error("No se encontró el modelo entrenado. Ejecuta primero el workflow de entrenamiento en GitHub Actions.")
    st.stop()


def preprocesar_para_prediccion(df_nuevo):
    train = pd.read_csv("application_train.csv")
    train[COLUMNA_ANOMALA] = train[COLUMNA_ANOMALA].replace(VALOR_ANOMALO, np.nan)
    if 'TARGET' not in train.columns:
        train['TARGET'] = 0

    test = df_nuevo.copy()
    if COLUMNA_ANOMALA in test.columns:
        test[COLUMNA_ANOMALA] = test[COLUMNA_ANOMALA].replace(VALOR_ANOMALO, np.nan)
    if 'TARGET' in test.columns:
        test = test.drop(columns=['TARGET'])

    ids = test['SK_ID_CURR'].values if 'SK_ID_CURR' in test.columns else np.arange(len(test))

    # Marcar test con TARGET=2 y concatenar manualmente
    test_marcado = test.copy()
    test_marcado['TARGET'] = 2
    df = pd.concat([train, test_marcado], axis=0, ignore_index=True)

    # Limpieza manual (igual que train_and_export.py)
    predictoras = df.drop(columns=['TARGET'])
    n_obs, n_attr = predictoras.shape
    obs_vacios = (predictoras.isnull().sum(axis=1) / n_attr) >= 0.5
    df = df.loc[~obs_vacios].copy()
    predictoras = df.drop(columns=['TARGET'])
    n_obs, n_attr = predictoras.shape
    attr_vacios = (predictoras.isnull().sum(axis=0) / n_obs) >= 0.6
    if attr_vacios.sum() > 0:
        cols = predictoras.columns[~attr_vacios].tolist() + ['TARGET']
        df = df[cols]

    # Preprocesamiento usando la misma función
    df_procesado = preprocesar_datos(df, umbral_categorica=UMBRAL_CATEGORICA, verbose=False)

    # Separar solo el test
    df_test = df_procesado[df_procesado['TARGET'] == 2].drop(columns=['TARGET']).copy()

    # Alinear columnas al modelo
    cols_modelo = modelo.feature_name() if tipo_mod == 'lgb' else modelo.feature_names
    for col in cols_modelo:
        if col not in df_test.columns:
            df_test[col] = np.nan
    df_test = df_test[cols_modelo].copy()

    return df_test, ids


def predecir(df_listo):
    """Ejecuta la predicción según el tipo de modelo."""
    if tipo_mod == 'lgb':
        cols_modelo = modelo.feature_name()
        for i, cats in enumerate(modelo.pandas_categorical):
            col = cols_modelo[i]
            df_listo[col] = pd.Categorical(
                df_listo[col].astype(str),
                categories=[str(c) for c in cats]
            )
        return modelo.predict(df_listo)
    else:
        dmat = xgb.DMatrix(df_listo, enable_categorical=True)
        return modelo.predict(dmat)


# ── Sidebar con info del modelo ──────────────
with st.sidebar:
    st.markdown("## Modelo actual")
    st.success(f"**{metadata['config_id']}** ({metadata['modelo']})")
    st.markdown(f"**Versión:** `{metadata['version']}`")
    st.markdown(f"**AUC-ROC:** `{metadata['roc_auc']}`")
    st.markdown(f"**F1-Score:** `{metadata['f1_score']}`")
    st.divider()
    st.markdown("*El modelo se actualiza automáticamente via GitHub Actions cuando se sube nuevo dataset con el mensaje* `[entrenar]`")

# ── Tabs principales ─────────────────────────
tab1, tab2 = st.tabs(["Predicción por lote (CSV)", "🔄 Reentrenar modelo"])

# ── TAB 1: Predicción por lote ───────────────
with tab1:
    st.markdown("### Sube un CSV con múltiples clientes")
    st.markdown("El archivo debe tener las mismas columnas que el dataset original de Kaggle.")

    archivo = st.file_uploader("Subir CSV", type="csv", key="pred")

    if archivo:
        df_pred = pd.read_csv(archivo)
        st.markdown(f"**Registros cargados:** {len(df_pred):,}")
        st.dataframe(df_pred.head(5), use_container_width=True)

        if st.button("🚀 Predecir todos los registros", use_container_width=True):
            try:
                with st.spinner("Preprocesando datos (puede tardar 1-2 min)..."):
                    df_listo, ids = preprocesar_para_prediccion(df_pred)
                    probs = predecir(df_listo)

                n = min(len(ids), len(probs))
                resultado = pd.DataFrame({
                    'SK_ID_CURR': ids[:n],
                    'probabilidad_incumplimiento': probs[:n],
                    'clasificacion': np.where(probs[:n] >= 0.5, 'ALTO RIESGO', 'BAJO RIESGO')
                })

                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total clientes", f"{len(resultado):,}")
                col2.metric("Alto riesgo", f"{(resultado['clasificacion']=='ALTO RIESGO').sum():,}")
                col3.metric("Bajo riesgo", f"{(resultado['clasificacion']=='BAJO RIESGO').sum():,}")

                st.dataframe(resultado.head(50), use_container_width=True)

                csv_resultado = resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Descargar resultados completos",
                    csv_resultado,
                    "predicciones.csv",
                    "text/csv"
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

    st.warning("El CSV debe tener la misma estructura que el dataset original de Kaggle, incluyendo la columna `TARGET`.")

    archivo_train = st.file_uploader("Subir nuevo CSV de entrenamiento", type="csv", key="retrain")

    if archivo_train:
        df_new = pd.read_csv(archivo_train)
        st.markdown(f"**Registros cargados:** {len(df_new):,}")

        if 'TARGET' not in df_new.columns:
            st.error("El CSV no tiene columna TARGET. Asegúrate de subir el dataset de entrenamiento.")
        else:
            conteo = df_new['TARGET'].value_counts()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total registros", f"{len(df_new):,}")
            col2.metric("Pagará (TARGET=0)", f"{conteo.get(0, 0):,}")
            col3.metric("Incumplirá (TARGET=1)", f"{conteo.get(1, 0):,}")

            st.dataframe(df_new.head(5), use_container_width=True)

            if st.button("🔄 Reemplazar datos y reentrenar", use_container_width=True):
                try:
                    with st.spinner("Guardando nuevo dataset..."):
                        df_new.to_csv("application_train.csv", index=False)

                    with st.spinner("Haciendo commit y push al repositorio..."):
                        token = os.getenv("PAT_TOKEN", "")
                        if not token:
                            st.error("No se encontró PAT_TOKEN en el contenedor. Revisa el docker-compose.yml.")
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
                                # 'git commit' devuelve 1 si no hay cambios: lo toleramos
                                if cmd[1] == "commit" and "nothing to commit" in salida:
                                    continue
                                raise RuntimeError(f"'{' '.join(cmd[:2])}' falló: {salida.strip()}")

                    st.success("✅ Dataset subido al repositorio. GitHub Actions está reentrenando el modelo automáticamente.")
                    st.info("🕐 El proceso tarda entre 35-40 minutos. Cuando termine, recarga la página para ver la nueva versión del modelo en el sidebar.")

                except Exception as e:
                    st.error(f"Error al hacer push al repositorio: {e}")