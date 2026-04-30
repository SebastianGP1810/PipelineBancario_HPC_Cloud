# PipelineBancario_HPC_Cloud
### Automatización del entrenamiento de modelos en el sector financiero

Repositorio del Grupo 2 — Curso de Computación de Alto Desempeño y Cloud Computing  
Universidad del Pacífico · 2026-I

## Integrantes

- Córdova Delgado, Marietha Kristeen Alexandra
- Guevara Peralta, Sebastian Antonio Valentino
- Medina Manrique, Diego Rodrigo
- Quiñones Vivas, Diego Alejandro

---

## ¿Qué hace este proyecto?

Implementa un pipeline de Machine Learning para predecir el riesgo de incumplimiento crediticio sobre el dataset [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/datasets/youngdaniel/loan-dataset), combinando **paralelismo de tareas** y **paralelismo de datos** para reducir el tiempo de entrenamiento frente a una ejecución secuencial tradicional.

---

## Dataset

| Característica | Valor |
|---|---|
| Fuente | Kaggle · Home Credit Default Risk |
| Registros (train) | 307,511 |
| Columnas iniciales | 122 |
| Variable objetivo | `TARGET` binaria (0 = paga, 1 = incumple) |
| Desbalance de clases | ~92% clase 0 / ~8% clase 1 (ratio 11.4:1) |

---

## Arquitectura del pipeline

El pipeline se divide en **5 fases**. Las fases 3 y 5 aplican paralelismo:

### Fase 1 — Carga y corrección de anomalías
Lee los archivos CSV y corrige el valor centinela `365,243` en `DAYS_EMPLOYED` (codifica personas sin empleo formal), reemplazándolo por `NaN`.

### Fase 2 — Limpieza de nulos excesivos
Une train y test, luego elimina filas con más del **50%** de valores nulos y columnas con más del **60%**. El dataset queda en 356,241 registros y 105 columnas.

### Fase 3 — Preprocesamiento ⚡ paralelo
Las tres tareas de imputación son independientes entre sí, por lo que se ejecutan en **3 hilos simultáneos** usando `joblib.Parallel`:

- **Hilo 1** — Variables categóricas: imputación con `"Desconocido"`, conversión a tipo `category`
- **Hilo 2** — Variables continuas: imputación por mediana + escalado con `RobustScaler`
- **Hilo 3** — Variables enteras: imputación por mediana + conversión a `Int64`

### Fase 4 — Split estratificado
División en train 80% (245,998 registros) y validación 20% (61,500 registros), preservando la proporción de clases con `stratify=y`.

### Fase 5 — Entrenamiento ⚡ paralelo en dos niveles
Se evalúan **8 configuraciones de hiperparámetros** (4 LightGBM + 4 XGBoost) en paralelo:

```python
joblib.Parallel(n_jobs=8, backend='threading')(tareas)
```

Dentro de cada modelo, **OpenMP** distribuye el cálculo de histogramas entre todos los núcleos del CPU (`n_jobs=-1`). Esto genera dos niveles combinados:

- **Paralelismo de tareas**: 8 modelos corriendo al mismo tiempo
- **Paralelismo de datos**: cada modelo usa todos los núcleos disponibles vía OpenMP

---

## Modelos y configuraciones evaluadas

| ID | Modelo | Hiperparámetros |
|---|---|---|
| LGB-1 | LightGBM | lr=0.05, num_leaves=31 |
| LGB-2 | LightGBM | lr=0.05, num_leaves=63 |
| LGB-3 | LightGBM | lr=0.10, num_leaves=31 |
| LGB-4 | LightGBM | lr=0.01, num_leaves=127 |
| XGB-1 | XGBoost | lr=0.05, max_depth=6 |
| XGB-2 | XGBoost | lr=0.05, max_depth=8 |
| XGB-3 | XGBoost | lr=0.10, max_depth=6 |
| XGB-4 | XGBoost | lr=0.01, max_depth=10 |

El desbalance de clases se maneja con `is_unbalance=True` en LightGBM y `scale_pos_weight` calculado dinámicamente en XGBoost.

---

## Resultados

### Desempeño predictivo (conjunto de validación)

| Config. | Modelo | AUC-ROC | F1-score |
|---|---|---|---|
| **LGB-1** | **LightGBM** | **0.7597** | **0.2730** |
| LGB-4 | LightGBM | 0.7594 | 0.2837 |
| LGB-2 | LightGBM | 0.7591 | 0.2780 |
| XGB-1 | XGBoost | 0.7557 | 0.2787 |

Las 4 configuraciones de LightGBM superan a las 4 de XGBoost en AUC-ROC.

### Desempeño computacional (speedup)

| Fase | Secuencial | Paralelo | Speedup |
|---|---|---|---|
| Carga | 6.94 s | 6.63 s | 1.05x |
| Limpieza | 3.24 s | 2.80 s | 1.16x |
| Preprocesamiento | 11.34 s | 8.17 s | 1.39x |
| Split | 1.30 s | 1.29 s | 1.01x |
| **Entrenamiento** | **864.44 s** | **370.93 s** | **2.33x** |
| **TOTAL** | **887.34 s** | **389.89 s** | **2.27x** |

El speedup global de **2.27x** se concentra en la fase de entrenamiento (2.33x). Las fases de carga y split están limitadas por I/O y no se benefician significativamente del paralelismo.

---

## Requisitos

- pandas
- numpy
- scikit-learn
- lightgbm>=4.6.0
- xgboost>=3.2.0
- joblib
- matplotlib

---

## Cómo ejecutar

1. Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/youngdaniel/loan-dataset) y coloca `application_train.csv` y `application_test.csv` en la misma carpeta que el notebook.
2. Abre `ML_bancario_3.1.ipynb` en Jupyter o Google Colab.
3. Ejecuta las celdas en orden. El notebook corre primero la versión secuencial y luego la paralela, reportando tiempos y speedup por fase al final.

---
