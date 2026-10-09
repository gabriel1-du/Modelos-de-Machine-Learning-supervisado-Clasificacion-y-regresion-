# EP3 — Machine Learning Supervisado: Clasificación y Regresión

**Asignatura:** TLY1101 — Machine Learning | **Sección:** 002D

Proyecto de Machine Learning supervisado desarrollado en `notebookDesarrollo.ipynb`. Incluye dos casos:

- **Caso 1 (Regresión):** predicción del precio de venta de autos usados (`car_data.csv`) con SVR y KNN.
- **Caso 2 (Clasificación):** predicción de cáncer de mama benigno/maligno (`breast_cancer.csv`) con Regresión Logística y Árboles de Decisión.

**Integrantes:** Martin Higuera y Gabriel Duran.

**Declaración de uso de IA:** se usaron herramientas de IA como apoyo para redacción, organización y revisión del notebook. Todo el código, los análisis y las justificaciones fueron revisados y validados por el equipo, conforme a la política institucional: https://bibliotecas.duoc.cl/ia

---

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `notebookDesarrollo.ipynb` | Notebook principal con ambos casos (213 celdas, ejecutado sin errores) |
| `data_sets/car_data.csv` | Dataset original del Caso 1 (301 registros) |
| `data_sets/breast_cancer.csv` | Dataset original del Caso 2 (569 registros) |
| `copy/car_data_copy.csv` | Copia de seguridad del CSV del Caso 1 |
| `copy/breast_can_copy.csv` | Copia de seguridad del CSV del Caso 2 |
| `Checklist` | Bitácora de avance del equipo |

Para reproducir: abrir `notebookDesarrollo.ipynb` y ejecutar todas las celdas en orden. Los CSV de respaldo en `copy/` permiten restaurar los datos originales si se alteran los de `data_sets/`.

---

## Caso 1: Regresión — Predicción del Precio de Venta de Autos Usados

**Objetivo:** predecir el **precio de venta de un auto usado** (target `precio_venta_actual`, el precio en que el vehículo se está vendiendo, tal como lo define el encargo) a partir de sus características: año, kilometraje, tipo de combustible, tipo de vendedor, transmisión, cantidad de propietarios y precio del vehículo nuevo.

El dataset (`car_data.csv`) contiene **301 registros**. El encabezado del archivo trae **8 nombres para filas de 9 valores** (falta el nombre de la cuarta columna, el precio actual del vehículo nuevo), por lo que se lee indicando explícitamente los **9 nombres correctos**; de lo contrario pandas desplaza todas las etiquetas una posición.

### Preprocesamiento (Caso 1)

| Paso | Resultado |
|---|---|
| Copia de seguridad | Se copió el CSV original a `copy/car_data_copy.csv`. |
| Corrección de encabezado | Lectura con los 9 nombres correctos; target `precio_venta_actual`. |
| `nom_auto` | **Eliminada**: 98 categorías distintas en 301 registros (1–2 apariciones por modelo), sin información generalizable. |
| Categóricas (`tipo_combustible`, `tipo_vendedor`, `transmision`) | **One-Hot Encoding** con `drop_first=True` (evita la trampa de las variables dummy). |
| Correlación | La más correlacionada con el precio de venta es `precio_actual`, seguida de `anno`; `kilometraje` correlaciona en forma negativa. |
| Partición | 80 % entrenamiento (**~240** registros) / 20 % prueba (**~61** registros), con `random_state=42`. |
| Escalamiento | `StandardScaler`, ajustado **solo con el conjunto de entrenamiento** (evita fuga de datos). Necesario porque SVR y KNN trabajan con distancias y las magnitudes difieren mucho (p. ej. `kilometraje` > 200.000 vs. `propietario` entre 0 y 3). |

### Modelos (Caso 1, sin GridSearchCV — el encargo no lo pide en este caso)

| Modelo | Configuración |
|---|---|
| SVR | `kernel='rbf'`, `C=1.0`, `epsilon=0.1` |
| KNN | `n_neighbors=7` |

Métricas: MSE, RMSE, MAE y R² (prueba y entrenamiento).

| Modelo | MSE | RMSE | MAE | R² prueba | R² entrenamiento |
|---|---|---|---|---|---|
| SVR (`rbf`, C=1.0, ε=0.1) | 5,1198 | 2,2627 | 1,0108 | 0,7777 | 0,6565 |
| **KNN (`n_neighbors=7`)** | **1,6512** | **1,2850** | **0,7790** | **0,9283** | 0,8906 |

**Mejor modelo de regresión:** KNN, con RMSE de 1,2850 y R² de 0,9283 en prueba, sin sobreajuste. El SVR quedó en subajuste con la configuración del encargo.

---

## Caso 2: Clasificación — Predicción de Cáncer de Mama (Benigno vs Maligno)

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
| **Regresión Logística (`class_weight="balanced"`)** | **98,25 %** | 100 % | **95,24 %** | **97,56 %** | **0,9980** | **2** | 0 |
| Árbol de Decisión (`max_depth=3`) | 91,23 % | 94,44 % | 80,95 % | 87,18 % | 0,8986 | 8 | 2 |
| Árbol de Decisión (sin restricción) | 92,98 % | 90,48 % | 90,48 % | 90,48 % | 0,9246 | 4 | 4 |

### Verificación de overfitting

| Modelo | Accuracy Train | Accuracy Test | Brecha |
|---|---|---|---|
| Regresión Logística | 96,92 % | 97,37 % | 0,45 % |
| Regresión Logística (balanced) | 97,58 % | 98,25 % | 0,66 % |
| Árbol (`max_depth=3`) | 96,70 % | 91,23 % | 5,48 % |
| Árbol (sin restricción) | 100,00 % | 92,98 % | **7,02 %** |

### Conclusiones

- La **Regresión Logística** fue la mejor familia de modelos: 97,37 % de accuracy (base) y 98,25 % (balanced), 0 falsos positivos y solo 3 (base) o 2 (balanced) falsos negativos, con un AUC de 0,9980.
- **GridSearchCV no mejoró el desempeño**: los mejores hiperparámetros (`max_iter=50`, `solver='newton-cg'`) dieron resultados idénticos al modelo base, por lo que el modelo original se mantiene.
- El **balanceo con `class_weight="balanced"`** aumentó el recall de la clase maligna (de 92,86 % a 95,24 %) y redujo los falsos negativos de 3 a 2, sin agregar falsos positivos. Dado que en un contexto oncológico los falsos negativos son el error más costoso, se selecciona esta variante como modelo final de clasificación.
- El **árbol sin restricción presenta overfitting**: logra 100 % en entrenamiento pero cae a 92,98 % en prueba (brecha de 7,02 %). La curva de validación confirma que el sobreajuste aparece al superar una profundidad de 3.
- Limitar la profundidad (`max_depth=3`) reduce la brecha de overfitting, pero el árbol sigue siendo inferior a la regresión logística.
- La debilidad principal de los árboles es el **recall de la clase maligna** (80,95 % y 90,48 %), ya que en un contexto oncológico los falsos negativos son el error más costoso.

---

## Selección final de modelos

| Caso | Modelo seleccionado | Desempeño clave en prueba |
|---|---|---|
| Caso 1 (Regresión) | **KNeighborsRegressor (`n_neighbors=7`)** | RMSE 1,2850 · R² 0,9283 · sin sobreajuste (R² train 0,8906) |
| Caso 2 (Clasificación) | **Regresión Logística con `class_weight="balanced"`** | Accuracy 98,25 % · Recall maligno 95,24 % · ROC-AUC 0,9980 · 2 FN, 0 FP |

Detalle de la comparación en el notebook: tabla resumen de regresión (sección 11 del Caso 1) y tabla "Comparación de modelos: Regresión Logística vs Árboles de Decisión" al final del Caso 2.

### Limitaciones y posibles mejoras

- **Datasets pequeños** (301 y 569 registros) con una única partición train/test; los resultados pueden variar con otra semilla. Una validación cruzada más robusta daría mayor confianza.
- **Multicolinealidad** en el Caso 2: las 30 variables miden características celulares muy relacionadas (radio, perímetro y área miden lo mismo en distintas escalas). Se conservaron todas por encargo; una selección de features o PCA podría simplificar el modelo.

---

## Librerías usadas

| Librería | Uso |
|---|---|
| `pandas` | Carga y manipulación del dataset |
| `matplotlib` | Gráficos |
| `seaborn` | Heatmaps, boxplots y matrices de confusión |
| `scikit-learn` | `train_test_split`, `StandardScaler`, `MinMaxScaler`, `SVR`, `KNeighborsRegressor`, `LogisticRegression`, `DecisionTreeClassifier`, `plot_tree`, `GridSearchCV` y métricas (`classification_report`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `roc_curve`, `roc_auc_score`, `mean_squared_error`, `mean_absolute_error`, `r2_score`) |
| `numpy` | Operaciones numéricas |
| `math` | Cálculo de la cuadrícula de gráficos |
| `shutil` | Copia del archivo original |

---

## Integrantes

- Gabriel Duran
- Martin Higuera

## GitHub

[@gabriel1-du](https://github.com/gabriel1-du)
