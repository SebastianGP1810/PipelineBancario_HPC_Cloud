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
                # Eliminar TARGET si existe
                if 'TARGET' in df_pred.columns:
                    df_pred = df_pred.drop(columns=['TARGET'])

                # Convertir columnas object a category
                for col in df_pred.select_dtypes(include='object').columns:
                    df_pred[col] = df_pred[col].astype('category')

                if tipo_mod == 'lgb':
                    # Usar solo las columnas que conoce el modelo
                    cols_modelo = modelo.feature_name()
                    cols_disponibles = [c for c in cols_modelo if c in df_pred.columns]
                    df_pred = df_pred[cols_disponibles]
                    probs = modelo.predict(df_pred)
                else:
                    dmat = xgb.DMatrix(df_pred, enable_categorical=True)
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

# ── TAB 2: Reentrenamiento ───────────────────
with tab2:
    st.markdown("### Reentrenar el modelo con nueva data")
    st.markdown("""
    Sube un CSV de entrenamiento ya particionado (solo registros de train con columna `TARGET`).
    El sistema reemplazará `application_train.csv`, disparará el reentrenamiento automático
    via GitHub Actions y actualizará el modelo desplegado.
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

            if st.button("Reemplazar datos y reentrenar", use_container_width=True):
                try:
                    with st.spinner("Guardando nuevo dataset..."):
                        df_new.to_csv("application_train.csv", index=False)

                    with st.spinner("Haciendo commit y push al repositorio..."):
                        import subprocess
                        subprocess.run(["git", "config", "user.name", "Streamlit Bot"], check=True)
                        subprocess.run(["git", "config", "user.email", "streamlit@bot.com"], check=True)
                        subprocess.run(["git", "add", "application_train.csv"], check=True)
                        subprocess.run(["git", "commit", "-m", "actualizar dataset de entrenamiento [entrenar]"], check=True)
                        subprocess.run(["git", "push"], check=True)

                    st.success("Dataset subido al repositorio. GitHub Actions está reentrenando el modelo automáticamente.")
                    st.info("El proceso tarda ~35-40 minutos. Cuando termine, recarga la página para ver la nueva versión del modelo en el sidebar.")

                except subprocess.CalledProcessError as e:
                    st.error(f"Error al hacer push al repositorio: {e}")
                except Exception as e:
                    st.error(f"Error inesperado: {e}")