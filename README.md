# Advanced Ensemble Learning Lab: Classification & Regression

Este repositorio es un laboratorio de Machine Learning enfocado en el uso de técnicas avanzadas de **Ensemble Learning** (Bagging, Boosting, Stacking y Cascading). 

El proyecto aborda dos problemas complejos utilizando conjuntos de datos con desafíos del mundo real: desequilibrio de clases, relaciones no lineales, colinealidad, presencia de outliers y ruido.

## Estructura del Repositorio

```text
Ensemble-Learning-Lab/
├── data/
│   ├── dataset_clasificacion.csv
│   └── dataset_regresion.csv
├── src/
│   ├── 01_classification_ensembles.py
│   └── 02_regression_ensembles.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Proyectos y Flujos de Trabajo

### 1. Clasificación Multiclase con Datos Desbalanceados
* **Script:** `src/01_classification_ensembles.py`
* **Desafío:** Clasificar una variable objetivo de 3 niveles con un desbalance significativo hacia la clase mayoritaria, además de lidiar con colinealidad  entre ciertas variables.
* **Pipeline aplicado:**
  * Tratamiento de colinealidad (eliminación manual basada en EDA).
  * Codificación mixta: *Ordinal Encoding* y *One-Hot Encoding*.
  * Oversampling mediante **SMOTE** para equilibrar el conjunto de entrenamiento.
  * Estandarización de variables (`StandardScaler`).
* **Modelos comparados:** Regresión Logística, Decision Tree, KNN, Naïve Bayes, Random Forest, Gradient Boosting, Stacking y una implementación personalizada de **Cascading**.

### 2. Regresión con Ruido y Outliers
* **Script:** `src/02_regression_ensembles.py`
* **Desafío:** Predecir una variable continua de alta varianza enfrentando valores atípicos severos (outliers) y variables irrelevantes.
* **Pipeline aplicado:**
  * Detección de outliers mediante rango intercuartílico (IQR).
  * Imputación avanzada de valores atípicos utilizando **KNN Imputer**.
  * Optimización exhaustiva de hiperparámetros mediante `GridSearchCV`.
* **Modelos comparados:** Regresión Lineal, Árboles de Decisión, Gradient Boosting Regressor y un **Stacking Regressor** (combinando enfoques paramétricos y no paramétricos con un meta-modelo Ridge).

## Insights y Resultados Destacados

1. **El poder del Gradient Boosting en Clasificación:** Frente a los modelos base (Accuracy ~56-69%), el **Gradient Boosting** optimizado logró un F1-Macro superior al **0.80**, demostrando su robustez para capturar interacciones no lineales sin sesgarse hacia la clase mayoritaria.
2. **Estrategias Secuenciales (Stacking vs Cascading):** Mientras que el Stacking funcionó de manera excelente en regresión al combinar sesgos inductivos diversos (lineales y basados en distancias), la arquitectura personalizada de Cascading demostró ser eficiente computacionalmente, aunque propensa a la propagación de errores en etapas tempranas si no se eligen los modelos adecuados.
3. **Manejo de Outliers en Regresión:** La sustitución de valores atípicos por imputaciones de vecindad (KNN) estabilizó el entrenamiento de la Regresión Lineal ($R^2$ ~0.50), sentando una base sólida que luego fue superada por el **Stacking Regressor** ($R^2$ ~0.54).

## Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone [https://github.com/Javier-MR-Leon/Ensemble-Learning-Lab.git](https://github.com/Javier-MR-Leon/Ensemble-Learning-Lab.git)
cd Ensemble-Learning-Lab
```

### 2. Entorno de Python
Se recomienda el uso de un entorno virtual (**Conda**) para garantizar la compatibilidad de las librerías.
* **Versión recomendada:** Python 3.10.19
* **Instalación de dependencias:**
```bash
pip install -r requirements.txt
```
