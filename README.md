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

**Features (30 variables predictoras):** todas las columnas de medición celular (ninguna se eliminó):

| Tipo | Variables |
|---|---|
| **Promedio (`_mean`)** | `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean` |
| **Error estándar (`_se`)** | `radius_se`, `texture_se`, `perimeter_se`, `area_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave points_se`, `symmetry_se`, `fractal_dimension_se` |
| **Peor valor (`_worst`)** | `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `smoothness_worst`, `compactness_worst`, `concavity_worst`, `concave points_worst`, `symmetry_worst`, `fractal_dimension_worst` |

Las variables más correlacionadas con el target fueron `concave points_worst` (0,79), `perimeter_worst` (0,78), `concave points_mean` (0,78) y `radius_worst` (0,78). Algunas variables (`smoothness_se`, `fractal_dimension_mean`, `texture_se`, `symmetry_se`) tienen correlación prácticamente nula con el target.

---

## Resultados de preprocesamiento de datos

| Paso | Resultado |
|---|---|
| Copia de seguridad | Se copió el CSV original a `copy/breast_can_copy.csv` para no alterar los datos originales. |
| Revisión de nulos | 0 nulos en todo el dataset, no se requirió imputación. |
| Tipos de datos | 30 `float64` y 2 `int64`; todas las variables son numéricas. |
| Eliminación de columnas | De 32 columnas se pasó a 31: solo se eliminó `id`. Se verificó que no existe la columna `Unnamed: 32` en este dataset. |
| Outliers | Se detectaron mediante boxplots, pero **se conservaron**: corresponden en gran parte a células malignas y eliminarlos reduciría el dataset de forma significativa. |
| Partición de datos | 80 % entrenamiento (**455** registros) / 20 % prueba (**114** registros), con `random_state=42` y `stratify=y` para mantener la proporción de clases. |
| Escalamiento | `MinMaxScaler`, ajustado **solo con el conjunto de entrenamiento** y aplicado luego al de prueba (evita fuga de datos). Se aplicó porque las variables tienen magnitudes muy distintas (p. ej. `area_worst` hasta 4254 vs. `smoothness_se` ≈ 0,03). |

---

## Trato de columnas

| Columna(s) | Acción | Justificación |
|---|---|---|
| `id` | Eliminada | Identificador del registro, no aporta información predictiva. |
| `Unnamed: 32` | Se verificó su existencia | En este dataset no está presente, no fue necesario eliminarla. |
| 30 columnas de medición (`_mean`, `_se`, `_worst`) | **Conservadas** como features | Todas aportan información sobre el tejido celular; en conjunto describen tamaño, textura y heterogeneidad de la masa. |
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
| Regresión Logística (base) | **97,37 %** | 100 % | 92,86 % | 96,30 % | **0,9980** | 3 | 0 |
| Regresión Logística + GridSearchCV | **97,37 %** | 100 % | 92,86 % | 96,30 % | **0,9980** | 3 | 0 |
| Árbol de Decisión (`max_depth=3`) | 91,23 % | 94,44 % | 80,95 % | 87,18 % | 0,8986 | 8 | 2 |
| Árbol de Decisión (sin restricción) | 92,98 % | 90,48 % | 90,48 % | 90,48 % | 0,9246 | 4 | 4 |

### Verificación de overfitting

| Modelo | Accuracy Train | Accuracy Test | Brecha |
|---|---|---|---|
| Regresión Logística | 96,92 % | 97,37 % | 0,45 % |
| Árbol (`max_depth=3`) | 96,70 % | 91,23 % | 5,48 % |
| Árbol (sin restricción) | 100,00 % | 92,98 % | **7,02 %** |

### Conclusiones

- La **Regresión Logística** fue el mejor modelo: 97,37 % de accuracy, 0 falsos positivos y solo 3 falsos negativos, con un AUC de 0,9980.
- **GridSearchCV no mejoró el desempeño**: los mejores hiperparámetros (`max_iter=50`, `solver='newton-cg'`) dieron resultados idénticos al modelo base, por lo que el modelo original se mantiene.
- El **árbol sin restricción presenta overfitting**: logra 100 % en entrenamiento pero cae a 92,98 % en prueba (brecha de 7,02 %). La curva de validación confirma que el sobreajuste aparece al superar una profundidad de 3.
- Limitar la profundidad (`max_depth=3`) reduce la brecha de overfitting, pero el árbol sigue siendo inferior a la regresión logística.
- La debilidad principal de los árboles es el **recall de la clase maligna** (80,95 % y 90,48 %), ya que en un contexto oncológico los falsos negativos son el error más costoso.

---

## Librerías usadas

| Librería | Uso |
|---|---|
| `pandas` | Carga y manipulación del dataset |
| `matplotlib` | Gráficos |
| `seaborn` | Heatmaps, boxplots y matrices de confusión |
| `scikit-learn` | `train_test_split`, `MinMaxScaler`, `LogisticRegression`, `DecisionTreeClassifier`, `plot_tree`, `GridSearchCV` y métricas (`classification_report`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `roc_curve`, `roc_auc_score`) |
| `numpy` | Operaciones numéricas |
| `math` | Cálculo de la cuadrícula de gráficos |
| `shutil` | Copia del archivo original |

---

## Integrantes

- Gabriel Duran
- Martin Higuera

## GitHub

[@gabriel1-du](https://github.com/gabriel1-du)
