import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import xgboost as xgb

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

# ── Sidebar con info del modelo ──────────────
with st.sidebar:
    st.markdown("## 📦 Modelo actual")
    st.success(f"**{metadata['config_id']}** ({metadata['modelo']})")
    st.markdown(f"**Versión:** `{metadata['version']}`")
    st.markdown(f"**AUC-ROC:** `{metadata['roc_auc']}`")
    st.markdown(f"**F1-Score:** `{metadata['f1_score']}`")
    st.divider()
    st.markdown("*El modelo se actualiza automáticamente via GitHub Actions cuando se sube nuevo dataset con el mensaje* `[entrenar]`")

# ── Tab principal ─────────────────────────
tab2, = st.tabs([ "📂 Predicción por lote (CSV)"])

# ── TAB 2: Predicción por lote ───────────────
with tab2:
    st.markdown("### Sube un CSV con múltiples clientes")
    st.markdown("El archivo debe tener las mismas columnas que el dataset original de Kaggle.")

    archivo = st.file_uploader("Subir CSV", type="csv")

    if archivo:
        df_pred = pd.read_csv(archivo)
        st.markdown(f"**Registros cargados:** {len(df_pred):,}")
        st.dataframe(df_pred.head(5), use_container_width=True)

        if st.button("🚀 Predecir todos los registros", use_container_width=True):
            for col in df_pred.select_dtypes(include='object').columns:
                df_pred[col] = df_pred[col].astype('category')

            try:
                if tipo_mod == 'lgb':
                    probs = modelo.predict(df_pred)
                else:
                    dmat  = xgb.DMatrix(df_pred, enable_categorical=True)
                    probs = modelo.predict(dmat)

                df_pred['probabilidad_incumplimiento'] = probs
                df_pred['clasificacion'] = np.where(probs >= 0.5, 'ALTO RIESGO', 'BAJO RIESGO')

                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total clientes", f"{len(df_pred):,}")
                col2.metric("Alto riesgo", f"{(df_pred['clasificacion']=='ALTO RIESGO').sum():,}")
                col3.metric("Bajo riesgo", f"{(df_pred['clasificacion']=='BAJO RIESGO').sum():,}")

                st.dataframe(
                    df_pred[['probabilidad_incumplimiento', 'clasificacion']].head(50),
                    use_container_width=True
                )

                csv_resultado = df_pred.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Descargar resultados completos",
                    csv_resultado,
                    "predicciones.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"Error al predecir: {e}")