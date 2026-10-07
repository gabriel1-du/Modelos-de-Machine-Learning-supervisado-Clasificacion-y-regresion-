# Predicción de Cáncer de Mama (Benigno vs Maligno)

Proyecto de Machine Learning supervisado desarrollado en `notebookDesarrollo.ipynb`. Se entrenan y comparan modelos de clasificación para predecir si un tumor de mama es **benigno** o **maligno** a partir de mediciones celulares.

---

## Contexto del caso

El cáncer de mama es un tipo de cáncer que se forma en las células de las mamas. Después del cáncer de piel, es el más comúnmente diagnosticado en mujeres a nivel mundial (aunque también puede presentarse en hombres). Gracias a la concientización, la financiación de investigaciones y la detección temprana, las tasas de supervivencia han aumentado y las muertes asociadas han disminuido de forma sostenida.

El dataset (`breast_cancer.csv`) contiene **mediciones celulares obtenidas de imágenes digitalizadas de una masa mamaria**. Cuenta con:

- **569 registros** (pacientes) y **32 columnas** originales.
- 30 variables numéricas (`float64`) de medición celular, más `id` y `diagnosis` (enteras).
- **0 valores nulos**.
- Cada característica celular se entrega en tres versiones: promedio (`_mean`), error estándar (`_se`) y peor valor (`_worst`).
- Distribución de clases desbalanceada hacia casos benignos: **62,74 % Benigno (0)** y **37,26 % Maligno (1)**.

El objetivo es construir un clasificador que apoye la detección temprana de tumores malignos, priorizando minimizar los **falsos negativos** (cáncer no detectado).

---

## Variables Target y Features

**Variable target:** `diagnosis`
- `0` → Benigno
- `1` → Maligno

**Features (24 variables predictoras finales):** mediciones celulares agrupadas en tres tipos:

| Tipo | Variables |
|---|---|
| **Promedio (`_mean`)** | `texture_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean` |
| **Error estándar (`_se`)** | `texture_se`, `area_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave points_se`, `symmetry_se`, `fractal_dimension_se` |
| **Peor valor (`_worst`)** | `texture_worst`, `area_worst`, `smoothness_worst`, `compactness_worst`, `concavity_worst`, `concave points_worst`, `symmetry_worst`, `fractal_dimension_worst` |

Las variables más correlacionadas con el target (dataset original) fueron `concave points_worst` (0,79), `perimeter_worst` (0,78), `concave points_mean` (0,78) y `radius_worst` (0,78). Algunas variables (`smoothness_se`, `fractal_dimension_mean`, `texture_se`, `symmetry_se`) tienen correlación prácticamente nula con el target.

---

## Resultados de preprocesamiento de datos

| Paso | Resultado |
|---|---|
| Copia de seguridad | Se copió el CSV original a `copy/breast_can_copy.csv` para no alterar los datos originales. |
| Revisión de nulos | 0 nulos en todo el dataset, no se requirió imputación. |
| Tipos de datos | 30 `float64` y 2 `int64`; todas las variables son numéricas. |
| Eliminación de columnas | De 32 columnas se pasó a 25 (24 features + target): se eliminaron `id` y 6 columnas de radio/perímetro. |
| Multicolinealidad (VIF) | Se redujo de valores extremos (`radius_mean` ≈ 3806, `perimeter_mean` ≈ 3786) a un máximo de ≈ 63,7 (`concavity_mean`). |
| Outliers | Se detectaron mediante boxplots, pero **se conservaron**: corresponden en gran parte a células malignas y eliminarlos reduciría el dataset de forma significativa. |
| Partición de datos | 80 % entrenamiento (**455** registros) / 20 % prueba (**114** registros), con `random_state=42` y `stratify=y` para mantener la proporción de clases. |
| Escalamiento | `MinMaxScaler`, ajustado **solo con el conjunto de entrenamiento** y aplicado luego al de prueba (evita fuga de datos). Se aplicó porque las variables tienen magnitudes muy distintas (p. ej. `area_worst` hasta 4254 vs. `smoothness_se` ≈ 0,03). |

---

## Trato de columnas

| Columna(s) | Acción | Justificación |
|---|---|---|
| `id` | Eliminada | Identificador del registro, no aporta información predictiva. |
| `Unnamed: 32` | Se verificó y eliminó en caso de existir | Columna vacía sin relevancia (en este dataset no estaba presente). |
| `radius_mean`, `radius_se`, `radius_worst` | Eliminadas | Colinealidad con el área. |
| `perimeter_mean`, `perimeter_se`, `perimeter_worst` | Eliminadas | Colinealidad con el área. |
| `area_*` | **Conservadas** | Radio, perímetro y área miden el tamaño de la masa; se dejó solo el área porque captura la magnitud total del espacio que ocupa la anomalía. |
| `*_se` (error estándar) | Conservadas | Capturan la heterogeneidad del tejido: el tejido benigno es uniforme y el maligno es más desordenado. |
| `*_worst` | Conservadas | Representan la media de los tres valores más extremos de cada característica, exponiendo la gravedad localizada del tejido. |
| `diagnosis` | Target | Variable a predecir. |

---

## Algoritmos y paradigmas usados

**Paradigma:** Aprendizaje supervisado, problema de **clasificación binaria**.

| Algoritmo | Configuración |
|---|---|
| **Regresión Logística (base)** | `solver='lbfgs'`, `max_iter=1000`, `random_state=42` |
| **Regresión Logística + GridSearchCV** | Búsqueda de `max_iter` ∈ {50, 100, 150, 200} y `solver` ∈ {newton-cg, lbfgs, liblinear, sag, saga}, con validación cruzada `cv=5` y métrica `accuracy`. |
| **Árbol de Decisión con restricción** | `max_depth=3`, `random_state=0` |
| **Árbol de Decisión sin restricción** | Sin límite de profundidad, `random_state=0` |

**Técnicas de evaluación:** Classification Report (precision, recall, F1), matriz de confusión, curva ROC y AUC, comparación train vs. test para detectar overfitting, y curva de validación por profundidad del árbol (`max_depth` de 1 a 9).

---

## Resultados del entrenamiento y conclusiones

### Comparación de modelos (conjunto de prueba, 114 registros)

| Modelo | Accuracy | Precision (Maligno) | Recall (Maligno) | F1 (Maligno) | ROC-AUC | FN | FP |
|---|---|---|---|---|---|---|---|
| Regresión Logística (base) | **98,25 %** | 100 % | 95,24 % | 97,56 % | 0,9974 | 2 | 0 |
| Regresión Logística + GridSearchCV | **98,25 %** | 100 % | 95,24 % | 97,56 % | — | 2 | 0 |
| Árbol de Decisión (`max_depth=3`) | 94,74 % | 100 % | 85,71 % | 92,31 % | 0,9869 | 6 | 0 |
| Árbol de Decisión (sin restricción) | 92,98 % | 92,50 % | 88,10 % | 90,24 % | 0,9196 | 5 | 3 |

### Verificación de overfitting

| Modelo | Accuracy Train | Accuracy Test | Brecha |
|---|---|---|---|
| Regresión Logística | 96,04 % | 98,25 % | 2,20 % |
| Árbol (`max_depth=3`) | 96,48 % | 94,74 % | 1,75 % |
| Árbol (sin restricción) | 100,00 % | 92,98 % | **7,02 %** |

### Conclusiones

- La **Regresión Logística** fue el mejor modelo: 98,25 % de accuracy, 0 falsos positivos y solo 2 falsos negativos, con un AUC de 0,9974.
- **GridSearchCV no mejoró el desempeño**: los mejores hiperparámetros (`max_iter=50`, `solver='newton-cg'`) dieron resultados idénticos al modelo base, por lo que el modelo original se mantiene.
- El **árbol sin restricción presenta overfitting**: logra 100 % en entrenamiento pero cae a 92,98 % en prueba (brecha de 7,02 %). La curva de validación confirma que el sobreajuste aparece al superar una profundidad de 3.
- Limitar la profundidad (`max_depth=3`) mejora la generalización del árbol (accuracy en prueba de 92,98 % a 94,74 %), pero sigue siendo inferior a la regresión logística.
- La debilidad crítica de los árboles es el **recall de la clase maligna** (85,71 % y 88,10 %), ya que en un contexto oncológico los falsos negativos son el error más costoso.
- La multicolinealidad se redujo de forma importante, aunque algunas variables (`concavity_mean`, `concave points_mean`) aún mantienen VIF elevado.

> **Nota sobre el notebook:** en la última celda de conclusiones del notebook se menciona que "el modelo sin restricción es el mejor". Sin embargo, los resultados obtenidos muestran lo contrario: el árbol con `max_depth=3` supera al árbol sin restricción en accuracy, recall de benignos y brecha train/test, y ambos son superados por la regresión logística.

---

## Librerías usadas

| Librería | Uso |
|---|---|
| `pandas` | Carga y manipulación del dataset |
| `matplotlib` | Gráficos |
| `seaborn` | Heatmaps, boxplots y matrices de confusión |
| `scikit-learn` | `train_test_split`, `MinMaxScaler`, `LogisticRegression`, `DecisionTreeClassifier`, `plot_tree`, `GridSearchCV` y métricas (`classification_report`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `roc_curve`, `roc_auc_score`) |
| `statsmodels` | Cálculo del Factor de Inflación de Varianza (VIF) |
| `numpy` | Operaciones numéricas |
| `math` | Cálculo de la cuadrícula de gráficos |
| `shutil` | Copia del archivo original |

---

## Integrantes

- Gabriel Duran
- Martin Higuera

## GitHub

[@gabriel1-du](https://github.com/gabriel1-du)
