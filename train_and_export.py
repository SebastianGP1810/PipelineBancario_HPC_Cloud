# train_and_export.py
# ============================================================
# Pipeline completo de entrenamiento y exportación del modelo
# Fiel al ml_bancario.ipynb — incluye análisis secuencial,
# análisis de escalabilidad [1,2,4 núcleos] y exportación pkl
# ============================================================

import os
import json
import time
import pickle
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # sin pantalla (para correr en servidor/CI)
import matplotlib.pyplot as plt
import joblib
import psutil

import lightgbm as lgb
import xgboost as xgb

from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────
RUTA_TRAIN       = os.getenv("RUTA_TRAIN", "application_train.csv")
RUTA_TEST        = os.getenv("RUTA_TEST",  "application_test.csv")
COLUMNA_ANOMALA  = "DAYS_EMPLOYED"
VALOR_ANOMALO    = 365243   # centinela: >1000 años de empleo = sin empleo formal
UMBRAL_CATEGORICA = 0.05

# Configuraciones de hiperparámetros (igual que el notebook)
configs_lgb = [
    {'config_id': 'LGB-1', 'learning_rate': 0.05, 'num_leaves': 31},
    {'config_id': 'LGB-2', 'learning_rate': 0.05, 'num_leaves': 63},
    {'config_id': 'LGB-3', 'learning_rate': 0.10, 'num_leaves': 31},
    {'config_id': 'LGB-4', 'learning_rate': 0.01, 'num_leaves': 127},
]
configs_xgb = [
    {'config_id': 'XGB-1', 'learning_rate': 0.05, 'max_depth': 6},
    {'config_id': 'XGB-2', 'learning_rate': 0.05, 'max_depth': 8},
    {'config_id': 'XGB-3', 'learning_rate': 0.10, 'max_depth': 6},
    {'config_id': 'XGB-4', 'learning_rate': 0.01, 'max_depth': 10},
]

# Núcleos a evaluar (igual que celda 16 del notebook)
configs_nucleos = sorted(set([1, 2, 4]))

# ─────────────────────────────────────────────
# FASE 1 — Carga y corrección de anomalías
# ─────────────────────────────────────────────
def cargar_y_corregir_datos(ruta_train, ruta_test, col_anomala=None,
                             valor_anomalo=None, verbose=True):
    """Carga los CSVs y reemplaza el valor centinela por NaN en la columna indicada."""
    if verbose:
        print("=" * 60)
        print(" FASE 1: CARGA DE DATOS Y CORRECCIÓN DE ANOMALÍAS")
        print("=" * 60)

    train = pd.read_csv(ruta_train)
    test  = pd.read_csv(ruta_test)

    if col_anomala and valor_anomalo is not None:
        # El valor 365243 en DAYS_EMPLOYED equivale a más de 1000 años de empleo.
        # Es un centinela que el dataset usa para codificar ausencia de empleo formal.
        train[col_anomala] = train[col_anomala].replace(valor_anomalo, np.nan)
        test[col_anomala]  = test[col_anomala].replace(valor_anomalo, np.nan)
        if verbose:
            print(f" Anomalía corregida en la columna '{col_anomala}'.")
    if verbose:
        print(f"Tamaño Train: {train.shape} | Tamaño Test: {test.shape}")
    return train, test


# ─────────────────────────────────────────────
# FASE 2 — Limpieza de nulos excesivos
# ─────────────────────────────────────────────
def limpiar_nulos_excesivos(train, test, umbral_obs=0.5,
                             umbral_attr=0.6, auto_eliminar=True, verbose=True):
    """Une los datasets y elimina filas/columnas con demasiados nulos."""
    if verbose:
        print("\n" + "=" * 60)
        print(" FASE 2: LIMPIEZA DE NULOS EXCESIVOS")
        print("=" * 60)

    train_copy = train.copy()
    test_copy  = test.copy()
    test_copy['TARGET'] = 2  # marcador para identificar filas de test después

    df = pd.concat([train_copy, test_copy], axis=0, ignore_index=True)

    predictoras = df.drop(columns=['TARGET'])
    n_obs, n_attr = predictoras.shape

    obs_muchos_vacios = (predictoras.isnull().sum(axis=1) / n_attr) >= umbral_obs
    if obs_muchos_vacios.sum() > 0 and auto_eliminar:
        df = df.loc[~obs_muchos_vacios]
        if verbose:
            print(f"Se eliminaron {obs_muchos_vacios.sum()} filas por exceso de nulos.")

    # Recalcular porque el shape cambió si se eliminaron filas
    predictoras = df.drop(columns=['TARGET'])
    n_obs, n_attr = predictoras.shape
    attr_muchos_vacios = (predictoras.isnull().sum(axis=0) / n_obs) >= umbral_attr
    if attr_muchos_vacios.sum() > 0 and auto_eliminar:
        cols_a_mantener = predictoras.columns[~attr_muchos_vacios].tolist() + ['TARGET']
        df = df[cols_a_mantener]
        if verbose:
            print(f"Se eliminaron {attr_muchos_vacios.sum()} columnas por exceso de nulos.")
    if verbose:
        print(f"Dimensiones tras limpieza: {df.shape}")
    return df


# ─────────────────────────────────────────────
# --- Bloques individuales de imputación ----
# ─────────────────────────────────────────────
def _imputar_categoricas(df, cols_categoricas):
    """Imputa categóricas con 'Desconocido' y las convierte a tipo category."""
    if not cols_categoricas:
        return pd.DataFrame()
    imputer_cat = SimpleImputer(strategy='constant', fill_value='Desconocido')
    bloque = pd.DataFrame(
        imputer_cat.fit_transform(df[cols_categoricas]),
        columns=cols_categoricas,
        index=df.index
    )
    # LightGBM y XGBoost esperan el tipo 'category' para su manejo nativo
    for col in cols_categoricas:
        bloque[col] = bloque[col].astype(str).astype('category')
    return bloque


def _imputar_y_escalar_continuas(df, cols_continuas):
    """Imputa continuas con mediana y escala con RobustScaler."""
    if not cols_continuas:
        return pd.DataFrame()
    # La mediana es más robusta que la media ante valores extremos
    # habituales en variables financieras (ingresos, montos de crédito)
    imputer_num = SimpleImputer(strategy='median')
    valores_imputados = imputer_num.fit_transform(df[cols_continuas])
    # RobustScaler normaliza usando el rango intercuartílico
    scaler = RobustScaler()
    valores_escalados = scaler.fit_transform(valores_imputados)
    return pd.DataFrame(valores_escalados, columns=cols_continuas, index=df.index)


def _imputar_enteras(df, cols_enteras):
    """Imputa enteras con mediana y fuerza tipo Int64."""
    if not cols_enteras:
        return pd.DataFrame()
    imputer_int = SimpleImputer(strategy='median')
    valores = imputer_int.fit_transform(df[cols_enteras])
    bloque = pd.DataFrame(valores, columns=cols_enteras, index=df.index)
    # Int64 con mayúscula (nullable integer) permite conservar NaN si hubiera
    return bloque.round().astype('Int64')


def _detectar_tipo_columna(serie, umbral_categorica=0.05):
    """
    Clasifica una columna numérica en 'categorica', 'entera' o 'continua'.

    Regla 1 — Cardinalidad relativa: si la proporción de valores únicos
    sobre el total es menor al umbral (5%), se trata como categórica.
    Regla 2 — Presencia de decimales: sin decimales → entera; con → continua.
    """
    valores_no_nulos = serie.dropna()
    if len(valores_no_nulos) == 0:
        return 'continua'
    cardinalidad = serie.nunique() / len(serie)
    if cardinalidad < umbral_categorica:
        return 'categorica'
    if (valores_no_nulos % 1 == 0).all():
        return 'entera'
    return 'continua'


def _clasificar_columnas_auto(df, umbral_categorica=0.05, verbose=True):
    """Clasifica automáticamente todas las columnas del dataframe."""
    cols_object   = df.select_dtypes(include=['object']).columns.tolist()
    cols_numericas = df.select_dtypes(include=['number']).columns.tolist()
    if 'TARGET' in cols_numericas:
        cols_numericas.remove('TARGET')

    cols_categoricas = list(cols_object)
    cols_enteras  = []
    cols_continuas = []

    for col in cols_numericas:
        tipo = _detectar_tipo_columna(df[col], umbral_categorica)
        if tipo == 'categorica':
            cols_categoricas.append(col)
        elif tipo == 'entera':
            cols_enteras.append(col)
        else:
            cols_continuas.append(col)

    if verbose:
        print(f"Detección automática: {len(cols_categoricas)} categóricas, "
              f"{len(cols_continuas)} continuas, {len(cols_enteras)} enteras")
    return cols_categoricas, cols_continuas, cols_enteras

# --- FASE 3: Preprocesamiento con detección automática ---
def preprocesar_datos(df_unido, umbral_categorica=0.05, verbose=True):
    """Imputa, escala y da formato a las variables — Fase 3 paralela."""
    if verbose:
        print("\n" + "=" * 60)
        print(" FASE 3: PREPROCESAMIENTO (PARALELO — 3 hilos)")
        print("=" * 60)

    df = df_unido.copy()
    cols_categoricas, cols_continuas, cols_enteras = _clasificar_columnas_auto(
        df, umbral_categorica=umbral_categorica, verbose=verbose
    )


    bloque_cat  = _imputar_categoricas(df, cols_categoricas)
    bloque_cont = _imputar_y_escalar_continuas(df, cols_continuas)
    bloque_int  = _imputar_enteras(df, cols_enteras)

    if not bloque_cat.empty:  df[cols_categoricas] = bloque_cat
    if not bloque_cont.empty: df[cols_continuas]   = bloque_cont
    if not bloque_int.empty:  df[cols_enteras]      = bloque_int

    return df


# ─────────────────────────────────────────────
# FASE 4 — Split estratificado
# ─────────────────────────────────────────────
def preparar_sets_entrenamiento(df_procesado, test_size=0.20,
                                 random_state=42, verbose=True):
    """Separa el dataset en train, validación y test final."""
    if verbose:
        print("\n" + "=" * 60)
        print(" FASE 4: SEPARACIÓN DE SETS DE DATOS")
        print("=" * 60)

    # Recuperar sets usando el marcador TARGET=2 del test
    train_limpio = df_procesado[df_procesado['TARGET'].isin([0, 1])].copy()
    test_limpio  = df_procesado[df_procesado['TARGET'] == 2].copy()

    X = train_limpio.drop(columns=['TARGET'])
    y = train_limpio['TARGET'].astype(int)
    X_test_final = test_limpio.drop(columns=['TARGET'])

    # stratify=y preserva la proporción de impagos (~8%) en train y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if verbose:
        print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test Final: {X_test_final.shape}")
    return X_train, X_val, y_train, y_val, X_test_final


# ─────────────────────────────────────────────
# FASE 5 — Funciones de entrenamiento
# ─────────────────────────────────────────────
def entrenar_lightgbm(X_train, y_train, X_val, y_val, config_id='LGB-default',
                       learning_rate=0.05, num_leaves=31,
                       num_boost_round=1000, n_jobs=-1):
    """Entrena LightGBM y devuelve modelo, métricas y tiempo."""
    t0 = time.time()
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'n_jobs': n_jobs,       # activa OpenMP: distribuye histogramas entre núcleos
        'is_unbalance': True,   # compensa el desbalance 11:1 internamente
        'verbose': -1
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)
    callbacks  = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)
    ]
    modelo_lgb = lgb.train(
        params, train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        callbacks=callbacks
    )
    y_pred_prob  = modelo_lgb.predict(X_val)
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    auc     = roc_auc_score(y_val, y_pred_prob)
    f1      = f1_score(y_val, y_pred_class)
    elapsed = time.time() - t0
    print(f"  [{config_id}] LightGBM | AUC: {auc:.4f} | F1: {f1:.4f} | Tiempo: {elapsed:.2f}s")
    return {
        'config_id': config_id, 'modelo': 'LightGBM', 'tipo': 'lgb',
        'objeto_modelo': modelo_lgb,
        'params': {'learning_rate': learning_rate, 'num_leaves': num_leaves},
        'roc_auc': auc, 'f1_score': f1, 'tiempo': elapsed
    }


def entrenar_xgboost(X_train, y_train, X_val, y_val, config_id='XGB-default',
                      learning_rate=0.05, max_depth=6,
                      num_boost_round=1000, scale_weight=1.0, n_jobs=-1):
    """Entrena XGBoost y devuelve modelo, métricas y tiempo."""
    t0 = time.time()
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',      # histogramas: más rápido que búsqueda exacta de splits
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'n_jobs': n_jobs,           # activa OpenMP internamente
        'scale_pos_weight': scale_weight,  # peso clase positiva para compensar desbalance
        'random_state': 42
    }
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval   = xgb.DMatrix(X_val,   label=y_val,   enable_categorical=True)
    modelo_xgb = xgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'entrenamiento'), (dval, 'validacion')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    y_pred_prob  = modelo_xgb.predict(dval)
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    auc     = roc_auc_score(y_val, y_pred_prob)
    f1      = f1_score(y_val, y_pred_class)
    elapsed = time.time() - t0
    print(f"  [{config_id}] XGBoost  | AUC: {auc:.4f} | F1: {f1:.4f} | Tiempo: {elapsed:.2f}s")
    return {
        'config_id': config_id, 'modelo': 'XGBoost', 'tipo': 'xgb',
        'objeto_modelo': modelo_xgb, 'dval': dval,
        'params': {'learning_rate': learning_rate, 'max_depth': max_depth},
        'roc_auc': auc, 'f1_score': f1, 'tiempo': elapsed
    }


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────
def main():
    print("\n" + "#" * 60)
    print("# INICIO DEL PIPELINE DE ENTRENAMIENTO Y EXPORTACIÓN")
    print("#" * 60)

    # ── Setup experimental — Hardware (igual que celda 13) ──
    print("\n" + "=" * 60)
    print(" SETUP EXPERIMENTAL — HARDWARE")
    print("=" * 60)
    print(f"  Procesador:        {platform.processor()}")
    print(f"  Núcleos físicos:   {psutil.cpu_count(logical=False)}")
    print(f"  Núcleos lógicos:   {psutil.cpu_count(logical=True)}")
    print(f"  RAM total:         {round(psutil.virtual_memory().total / 1e9, 1)} GB")
    print(f"  Sistema operativo: {platform.system()} {platform.release()}")
    print(f"  Python:            {platform.python_version()}")
    print(f"  LightGBM:          {lgb.__version__}")
    print(f"  XGBoost:           {xgb.__version__}")

    
    tiempos_secuencial = {}
    t_total_seq_start = time.time()
    
    # ── Fases 1 a 4 ─────────────────────────────────────────
    df_train_raw, df_test_raw = cargar_y_corregir_datos(
        RUTA_TRAIN, RUTA_TEST, COLUMNA_ANOMALA, VALOR_ANOMALO
    )
    df_unificado = limpiar_nulos_excesivos(df_train_raw, df_test_raw, auto_eliminar=True)
    df_listo     = preprocesar_datos(df_unificado, umbral_categorica=UMBRAL_CATEGORICA)
    X_train, X_val, y_train, y_val, X_test_final = preparar_sets_entrenamiento(df_listo)

    peso_clase_positiva = sum(y_train == 0) / sum(y_train == 1)

    print("\n" + "#" * 60)
    print("# EJECUCIÓN SECUENCIAL")
    print("#" * 60)

    tiempos_secuencial = {}
    t_total_seq_start = time.time()

    # --- Carga ---
    t0 = time.time()
    df_train_raw, df_test_raw = cargar_y_corregir_datos(
        RUTA_TRAIN, RUTA_TEST, COLUMNA_ANOMALA, VALOR_ANOMALO
    )
    tiempos_secuencial['Carga'] = time.time() - t0

    # --- Limpieza ---
    t0 = time.time()
    df_unificado = limpiar_nulos_excesivos(df_train_raw, df_test_raw, auto_eliminar=True)
    tiempos_secuencial['Limpieza'] = time.time() - t0

    # --- Preprocesamiento ---
    t0 = time.time()
    df_listo = preprocesar_datos(df_unificado, umbral_categorica=UMBRAL_CATEGORICA)
    tiempos_secuencial['Preprocesamiento'] = time.time() - t0

    # --- Split ---
    t0 = time.time()
    X_train, X_val, y_train, y_val, X_test_final = preparar_sets_entrenamiento(df_listo)
    tiempos_secuencial['Split'] = time.time() - t0

    # El peso de la clase positiva es n_negativos / n_positivos
    peso_clase_positiva = sum(y_train == 0) / sum(y_train == 1)

    # --- Entrenamiento con n_jobs=1 ---
    print("\n" + "=" * 60)
    print(" FASE 5 (SECUENCIAL): ENTRENAMIENTO DE 8 MODELOS")
    print("=" * 60)
    t0 = time.time()
    resultados_secuencial = []

    for cfg in configs_lgb:
        res = entrenar_lightgbm(X_train, y_train, X_val, y_val, n_jobs=1, **cfg)
        resultados_secuencial.append(res)

    for cfg in configs_xgb:
        res = entrenar_xgboost(X_train, y_train, X_val, y_val,
                            scale_weight=peso_clase_positiva, n_jobs=1, **cfg)
        resultados_secuencial.append(res)

    tiempos_secuencial['Entrenamiento'] = time.time() - t0
    tiempos_secuencial['TOTAL'] = time.time() - t_total_seq_start

    print("\n" + "=" * 60)
    print(" TIEMPOS SECUENCIALES")
    print("=" * 60)
    for fase, t in tiempos_secuencial.items():
        print(f"  {fase:<20s} {t:>8.2f}s")


    # ── Análisis de escalabilidad [1, 2, 4] (igual que celda 16) ──
    print(f"\nConfiguraciones a evaluar: {configs_nucleos}")
    resultados_escalabilidad = {}

    for n_jobs_config in configs_nucleos:
        print("\n" + "=" * 60)
        print(f" ENTRENAMIENTO CON n_jobs={n_jobs_config}")
        print("=" * 60)
        t0 = time.time()
        resultados_iter = []
        for cfg in configs_lgb:
            res = entrenar_lightgbm(X_train, y_train, X_val, y_val,
                                    n_jobs=n_jobs_config, **cfg)
            resultados_iter.append(res)
        for cfg in configs_xgb:
            res = entrenar_xgboost(X_train, y_train, X_val, y_val,
                                   scale_weight=peso_clase_positiva,
                                   n_jobs=n_jobs_config, **cfg)
            resultados_iter.append(res)
        tiempo_total = time.time() - t0
        resultados_escalabilidad[n_jobs_config] = {
            'tiempo_entrenamiento': tiempo_total,
            'resultados': resultados_iter
        }
        print(f"  Tiempo total con n_jobs={n_jobs_config}: {tiempo_total:.2f}s")

    n_jobs_vals = list(resultados_escalabilidad.keys())

    # ── Resultados de escalabilidad (igual que celda 18) ────
    t_base = resultados_escalabilidad[1]['tiempo_entrenamiento']
    print("\n" + "=" * 60)
    print(" RESULTADOS DE ESCALABILIDAD")
    print("=" * 60)
    print(f"{'n_jobs':<10} {'Tiempo (s)':>12} {'Speedup':>10} {'Speedup ideal':>14}")
    print("-" * 50)
    for n, datos in resultados_escalabilidad.items():
        t       = datos['tiempo_entrenamiento']
        speedup = t_base / t
        print(f"  {n:<8} {t:>12.2f}s {speedup:>10.2f}x {float(n):>13.1f}x")

    print("\n" + "=" * 60)
    print(" MÉTRICAS PREDICTIVAS (LGB-1, verificación de reproducibilidad)")
    print("=" * 60)
    print(f"{'n_jobs':<10} {'AUC-ROC':>10} {'F1-score':>10}")
    print("-" * 35)
    for n, datos in resultados_escalabilidad.items():
        primer = datos['resultados'][0]
        print(f"  {n:<8} {primer['roc_auc']:>10.4f} {primer['f1_score']:>10.4f}")

    # ── Tabla combinada AUC-ROC x Speedup x n_jobs (celda 20) ──
    filas = []
    for n in n_jobs_vals:
        t_total_n = resultados_escalabilidad[n]['tiempo_entrenamiento']
        speedup_n = t_base / t_total_n
        for res in resultados_escalabilidad[n]['resultados']:
            t_ref_cfg = next(
                r['tiempo'] for r in resultados_escalabilidad[1]['resultados']
                if r['config_id'] == res['config_id']
            )
            speedup_cfg = t_ref_cfg / res['tiempo'] if res['tiempo'] > 0 else 0
            filas.append({
                'config_id': res['config_id'], 'modelo': res['modelo'],
                'n_jobs': n, 'AUC-ROC': res['roc_auc'], 'F1-score': res['f1_score'],
                'tiempo (s)': round(res['tiempo'], 2), 'speedup': round(speedup_cfg, 2),
            })

    df_tabla = pd.DataFrame(filas)
    auc_norm     = (df_tabla['AUC-ROC'] - df_tabla['AUC-ROC'].min()) / \
                   (df_tabla['AUC-ROC'].max() - df_tabla['AUC-ROC'].min() + 1e-9)
    speedup_norm = (df_tabla['speedup'] - df_tabla['speedup'].min()) / \
                   (df_tabla['speedup'].max() - df_tabla['speedup'].min() + 1e-9)
    df_tabla['score_balance'] = ((auc_norm + speedup_norm) / 2).round(4)
    df_tabla = df_tabla.sort_values(['score_balance', 'AUC-ROC'],
                                     ascending=False).reset_index(drop=True)

    print("\n" + "=" * 85)
    print(" TABLA COMBINADA: AUC-ROC x SPEEDUP x n_jobs")
    print("=" * 85)
    print(df_tabla.to_string(index=False))

    optimo = df_tabla.iloc[0]
    print("\n" + "=" * 85)
    print(" CONFIGURACIÓN ÓPTIMA (mayor balance calidad / costo computacional)")
    print("=" * 85)
    print(f"  Modelo:      {optimo['config_id']} ({optimo['modelo']})")
    print(f"  n_jobs:      {int(optimo['n_jobs'])}")
    print(f"  AUC-ROC:     {optimo['AUC-ROC']:.4f}")
    print(f"  F1-score:    {optimo['F1-score']:.4f}")
    print(f"  Tiempo:      {optimo['tiempo (s)']:.2f}s")
    print(f"  Speedup:     {optimo['speedup']:.2f}x")
    print(f"  Score bal.:  {optimo['score_balance']:.4f}")

    # ── Gráficos de escalabilidad (celda 22) ────────────────
    tiempos_esc      = [resultados_escalabilidad[n]['tiempo_entrenamiento'] for n in n_jobs_vals]
    speedups_reales  = [t_base / t for t in tiempos_esc]
    speedups_ideales = [float(n) for n in n_jobs_vals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(n_jobs_vals, tiempos_esc, marker='o', color='#E07A5F', linewidth=2, label='Tiempo real')
    ax1.set_xlabel('Número de núcleos (n_jobs)')
    ax1.set_ylabel('Tiempo de entrenamiento (segundos)')
    ax1.set_title('Tiempo de entrenamiento vs núcleos')
    ax1.set_xticks(n_jobs_vals)
    ax1.grid(alpha=0.3)
    ax1.legend()
    for x, y in zip(n_jobs_vals, tiempos_esc):
        ax1.text(x, y + max(tiempos_esc) * 0.02, f'{y:.1f}s', ha='center', fontsize=9)

    ax2 = axes[1]
    ax2.plot(n_jobs_vals, speedups_reales,  marker='o', color='#3D5A80', linewidth=2, label='Speedup real')
    ax2.plot(n_jobs_vals, speedups_ideales, marker='s', color='#aaaaaa', linewidth=1.5,
             linestyle='--', label='Speedup ideal (lineal)')
    ax2.set_xlabel('Número de núcleos (n_jobs)')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Curva de escalabilidad')
    ax2.set_xticks(n_jobs_vals)
    ax2.grid(alpha=0.3)
    ax2.legend()
    for x, y in zip(n_jobs_vals, speedups_reales):
        ax2.text(x, y + max(speedups_ideales) * 0.02, f'{y:.2f}x', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('escalabilidad.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Gráfico guardado: escalabilidad.png")

    # ── Gráfico tiempo por modelo 1 núcleo vs máx (celda 23) ──
    resultados_1   = resultados_escalabilidad[1]['resultados']
    resultados_max = resultados_escalabilidad[max(configs_nucleos)]['resultados']
    df_1   = pd.DataFrame(resultados_1)[['config_id', 'tiempo']].rename(columns={'tiempo': 'n_jobs=1'})
    df_max = pd.DataFrame(resultados_max)[['config_id', 'tiempo']].rename(
                columns={'tiempo': f'n_jobs={max(configs_nucleos)}'})
    df_comp = df_1.merge(df_max, on='config_id')

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df_comp))
    ancho = 0.35
    ax.bar(x - ancho/2, df_comp['n_jobs=1'], ancho, label='n_jobs=1', color='#E07A5F')
    ax.bar(x + ancho/2, df_comp[f'n_jobs={max(configs_nucleos)}'], ancho,
           label=f'n_jobs={max(configs_nucleos)}', color='#3D5A80')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comp['config_id'])
    ax.set_ylabel('Tiempo de entrenamiento (segundos)')
    ax.set_title('Tiempo por configuración: 1 núcleo vs todos los disponibles')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('tiempo_por_modelo.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Gráfico guardado: tiempo_por_modelo.png")

    # ── Escalabilidad del mejor modelo (celda 24) ───────────
    df_sec          = pd.DataFrame(resultados_secuencial)
    mejor_config_id = df_sec.loc[df_sec['roc_auc'].idxmax(), 'config_id']
    print(f"\nMejor modelo: {mejor_config_id}")

    tiempos_mejor_modelo = []
    for n in n_jobs_vals:
        df_n    = pd.DataFrame(resultados_escalabilidad[n]['resultados'])
        t_mejor = df_n.loc[df_n['config_id'] == mejor_config_id, 'tiempo'].values[0]
        tiempos_mejor_modelo.append(t_mejor)

    t_base_mejor      = tiempos_mejor_modelo[0]
    speedups_mejor    = [t_base_mejor / t for t in tiempos_mejor_modelo]
    speedups_id_mejor = [float(n) for n in n_jobs_vals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.plot(n_jobs_vals, tiempos_mejor_modelo, marker='o', color='#E07A5F', linewidth=2)
    ax1.set_xlabel('Número de núcleos (n_jobs)')
    ax1.set_ylabel('Tiempo de entrenamiento (segundos)')
    ax1.set_title(f'Tiempo del mejor modelo ({mejor_config_id}) vs núcleos')
    ax1.set_xticks(n_jobs_vals)
    ax1.grid(alpha=0.3)
    for x, y in zip(n_jobs_vals, tiempos_mejor_modelo):
        ax1.text(x, y + max(tiempos_mejor_modelo) * 0.02, f'{y:.1f}s', ha='center', fontsize=9)

    ax2 = axes[1]
    ax2.plot(n_jobs_vals, speedups_mejor,   marker='o', color='#3D5A80', linewidth=2, label='Speedup real')
    ax2.plot(n_jobs_vals, speedups_id_mejor, marker='s', color='#aaaaaa', linewidth=1.5,
             linestyle='--', label='Speedup ideal (lineal)')
    ax2.set_xlabel('Número de núcleos (n_jobs)')
    ax2.set_ylabel('Speedup')
    ax2.set_title(f'Escalabilidad del mejor modelo ({mejor_config_id})')
    ax2.set_xticks(n_jobs_vals)
    ax2.grid(alpha=0.3)
    ax2.legend()
    for x, y in zip(n_jobs_vals, speedups_mejor):
        ax2.text(x, y + max(speedups_id_mejor) * 0.02, f'{y:.2f}x', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('escalabilidad_mejor_modelo.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Gráfico guardado: escalabilidad_mejor_modelo.png")

    # ── Exportar el mejor modelo como .pkl ──────────────────
    # Obtener el objeto modelo del mejor resultado del análisis de escalabilidad
    # usando el n_jobs óptimo según score_balance
    n_jobs_optimo = int(optimo['n_jobs'])
    mejor_resultado = next(
        r for r in resultados_escalabilidad[n_jobs_optimo]['resultados']
        if r['config_id'] == optimo['config_id']
    )

    version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Guardar modelo con versión y como modelo_actual (el que lee Streamlit)
    with open("modelo_actual.pkl", "wb") as f:
        pickle.dump({
            'modelo': mejor_resultado['objeto_modelo'],
            'tipo':   mejor_resultado['tipo']
        }, f)

    with open("modelo_metadata.json", "w") as f:
        json.dump({
            "version":   version,
            "config_id": mejor_resultado['config_id'],
            "modelo":    mejor_resultado['modelo'],
            "n_jobs":    n_jobs_optimo,
            "roc_auc":   round(mejor_resultado['roc_auc'], 4),
            "f1_score":  round(mejor_resultado['f1_score'], 4),
            "params":    mejor_resultado['params']
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(" MODELO EXPORTADO")
    print("=" * 60)
    print(f"  Archivo:   modelo_actual.pkl")
    print(f"  Versión:   {version}")
    print(f"  Config:    {mejor_resultado['config_id']} ({mejor_resultado['modelo']})")
    print(f"  n_jobs:    {n_jobs_optimo}")
    print(f"  AUC-ROC:   {mejor_resultado['roc_auc']:.4f}")
    print(f"  F1-score:  {mejor_resultado['f1_score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
