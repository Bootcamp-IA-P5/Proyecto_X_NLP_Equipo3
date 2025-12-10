# 🛡️ Proyecto X NLP - Detección de Mensajes de Odio en YouTube

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-BERT-yellow.svg)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-red.svg)](https://mlflow.org/)

## 📋 Descripción

Sistema de **detección automática de mensajes de odio** en comentarios de YouTube utilizando técnicas de **Procesamiento de Lenguaje Natural (NLP)** y **Machine Learning**. Este proyecto aborda la necesidad de YouTube de moderar contenido de manera eficiente, dado que los moderadores humanos no pueden manejar el creciente volumen de comentarios.

### 🎯 Objetivo

Crear modelos de Machine Learning capaces de identificar automáticamente comentarios de odio para permitir acciones como eliminación automática o baneo de usuarios. 

## 🏗️ Estructura del Proyecto

```
Proyecto_X_NLP_Equipo3/
├── 📁 . github/              # Configuración de GitHub (CI/CD, templates)
├── 📁 data/                 # Datos del proyecto
│   ├── processed/           # Datos procesados listos para modelado
│   └── raw/                 # Datos originales
├── 📁 frontend/             # Interfaz de usuario
│   └── src/                 # Código fuente del frontend
├── 📁 models/               # Modelos entrenados (. pkl)
├── 📁 notebooks/            # Jupyter notebooks de experimentación
│   ├── random_forest_hate_detection.ipynb
│   ├── XGBoostEnsemble.ipynb
│   ├── TransformersModel.ipynb
│   └── TrasformersModel2.ipynb
├── 📁 tests/                # Tests unitarios y de integración
├── 📄 demo_hate_detection. py  # Demo interactivo del sistema
├── 📄 requirements.txt      # Dependencias del proyecto
└── 📄 README.md
```

## 🤖 Modelos Implementados

| Modelo | Descripción | Características |
|--------|-------------|-----------------|
| **Random Forest** | Modelo ensemble basado en árboles de decisión | TF-IDF + Features numéricas, SMOTE para balanceo |
| **XGBoost Ensemble** | Ensemble de múltiples XGBoost | Data Augmentation, múltiples seeds, Feature engineering avanzado |
| **BERT Transformers** | Modelo de lenguaje pre-entrenado | Fine-tuning para clasificación de hate speech |

## 📊 Dataset

- **Tamaño**: ~997 comentarios de YouTube
- **Balance de clases**:
  - Normal (0): 538 (54.0%)
  - Odio (1): 459 (46.0%)

## 🚀 Instalación

### Prerrequisitos

- Python 3.8+
- pip o conda

### Pasos

1. **Clonar el repositorio**
```bash
git clone https://github.com/Bootcamp-IA-P5/Proyecto_X_NLP_Equipo3.git
cd Proyecto_X_NLP_Equipo3
```

2. **Crear entorno virtual** (recomendado)
```bash
python -m venv . venv
source . venv/bin/activate  # Linux/Mac
# o
. venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Descargar recursos de NLTK**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## 💻 Uso

### Demo Interactivo

Ejecuta el demo para probar el modelo con tus propios comentarios: 

```bash
python demo_hate_detection. py
```

**Opciones disponibles:**
1. **Demo con ejemplos predefinidos**:  Analiza comentarios de muestra
2. **Demo interactivo**: Ingresa tus propios comentarios para clasificar

### Ejemplo de Uso Programático

```python
from demo_hate_detection import load_model_components, predict_hate_comment

# Cargar modelo
model, tfidf, scaler, metadata = load_model_components()

# Predecir
comment = "This video is amazing, great content!"
result = predict_hate_comment(comment, model, tfidf, scaler)

print(f"Es odio: {result['is_hate']}")
print(f"Probabilidad: {result['hate_probability']:.1%}")
print(f"Confianza: {result['confidence']}")
```

# 🖥️ Interfaz Web con Streamlit

El proyecto incluye una **aplicación web interactiva** construida con Streamlit que permite probar todos los modelos de detección de odio de forma visual. 

## 🚀 Ejecutar la Aplicación

```bash
cd frontend/src
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`

## 📑 Funcionalidades

La interfaz está organizada en **4 pestañas**:

| Pestaña | Descripción |
|---------|-------------|
| 📊 **Información del Dataset** | Visualiza estadísticas del dataset, distribución de clases y muestra de datos |
| 📈 **Análisis de Modelos** | Lista todos los modelos cargados, sus parámetros y permite pruebas rápidas |
| 🔍 **Predicción en Vivo** | Ingresa cualquier texto y obtén la predicción en tiempo real |
| 🎬 **Predicción por Vídeo** | Analiza todos los comentarios de un vídeo específico del dataset |

## 🤖 Modelos Soportados

La aplicación detecta y carga automáticamente los siguientes tipos de modelos desde la carpeta `models/`:

| Tipo de Modelo | Formato Esperado |
|----------------|------------------|
| **XGBoost** | Diccionario con `model`, `vectorizer`, `scaler`, `threshold` |
| **Random Forest** | Diccionario con `model`, `vectorizer`, `scaler` |
| **BERT/Transformers** | Diccionario con `model`, `tokenizer`, `max_len` |
| **sklearn genérico** | Objeto con método `predict()` |

## 📸 Características

- ✅ **Carga automática** de todos los modelos `.pkl` disponibles
- ✅ **Comparación de modelos** - prueba el mismo texto con diferentes modelos
- ✅ **Visualización de probabilidades** por clase
- ✅ **Análisis masivo** de comentarios por vídeo
- ✅ **Panel lateral** con información del estado del sistema
- ✅ **Caché de recursos** para carga rápida de modelos y datos

## 🔍 Pestaña:  Predicción en Vivo

### Uso

1. Selecciona un modelo del dropdown
2. Escribe o pega el texto a analizar
3. Haz clic en "🔍 Analizar Mensaje"
4. Observa el resultado y las probabilidades por clase

### Ejemplo de Resultado

```
⚠️ HATE MESSAGE DETECTED
Confianza: 87.3%

Probabilidades: 
- No hate: 12.7%
- Hate: 87.3%
```

## 🎬 Pestaña: Predicción por Vídeo

Permite analizar **todos los comentarios** de un vídeo específico del dataset:

1. Selecciona la versión del dataset
2. Elige un `VideoId` del dropdown
3. Selecciona el modelo a utilizar
4. Haz clic en "Analizar todos los comentarios del vídeo"
5. Visualiza los resultados en tabla y gráfico de distribución

## 💡 Ejemplos de Mensajes para Probar

### Mensajes Normales
```
I really like this video, very informative. 
Thank you for sharing this content.
Great explanation, this helped me understand the topic. 
```

### Mensajes de Odio Potencial
```
You are such an idiot, you know nothing.
Get out of here, nobody wants you.
This is the worst content ever, you should quit.
```

## ⚙️ Configuración

### Variables a Ajustar en `app.py`

```python
# Carpeta donde se encuentran los modelos . pkl
MODELS_DIR = "models"

# Ruta al dataset procesado
DATASET_PATH = "data/processed/youtube_all_versions.pkl"
```

### Estructura Esperada

```
Proyecto_X_NLP_Equipo3/
├── frontend/
│   └── src/
│       └── app. py          # Aplicación Streamlit
├── models/
│   ├── random_forest_hate_model.pkl
│   ├── xgboost_hate_model.pkl
│   └── hate_speech_bert_*. pkl
└── data/
    └── processed/
        └── youtube_all_versions.pkl
```

## 📦 Dependencias Adicionales

Asegúrate de tener instaladas las siguientes librerías:

```bash
pip install streamlit pandas numpy joblib scipy
```

Para modelos BERT/Transformers: 
```bash
pip install torch transformers
```

## 🐛 Solución de Problemas

| Problema | Solución |
|----------|----------|
| "No hay modelos cargados" | Verifica que existan archivos `.pkl` en la carpeta `models/` |
| "Dataset no cargado" | Ajusta la variable `DATASET_PATH` con la ruta correcta |
| Error con BERT | Asegúrate de tener `torch` y `transformers` instalados |
| Predicción lenta | Los modelos BERT son más lentos; considera usar XGBoost/Random Forest para pruebas rápidas |

## 🔗 Enlaces Relacionados

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Código fuente:  frontend/src/app.py](https://github.com/Bootcamp-IA-P5/Proyecto_X_NLP_Equipo3/blob/dev/frontend/src/app.py)

## 🐳 Docker (En desarrollo)

La rama `26-creating-containers-docker-compose-and-docker-file` contiene la configuración de Docker para containerizar la aplicación.

```bash
# Próximamente
docker-compose up
```

## 📈 Pipeline de Procesamiento

1. **Preprocesamiento de texto**: 
   - Eliminación de URLs, menciones y hashtags
   - Normalización (lowercase, eliminación de números)
   - Eliminación de stopwords
   - Lematización

2. **Feature Engineering**:
   - Vectorización TF-IDF
   - Features numéricas:  longitud del texto, conteo de palabras, signos de exclamación, ratio de mayúsculas, etc.

3. **Balanceo de clases**:  SMOTE (Synthetic Minority Over-sampling Technique)

4. **Entrenamiento y evaluación** con cross-validation estratificado

## 📊 Métricas de Evaluación

El proyecto prioriza **Recall** para minimizar falsos negativos (no perder mensajes de odio):

- **F1-Score**: Balance entre precisión y recall
- **Recall**: Capacidad de detectar todos los mensajes de odio
- **Precision**:  Evitar falsos positivos
- **ROC-AUC**: Rendimiento general del clasificador

## 🔧 Dependencias Principales

| Librería | Uso |
|----------|-----|
| `scikit-learn` | Modelos ML, métricas, pipelines |
| `imbalanced-learn` | Técnicas de balanceo (SMOTE) |
| `xgboost` | Modelo XGBoost |
| `transformers` | Modelos BERT |
| `nltk` | Procesamiento de lenguaje natural |
| `pandas` / `numpy` | Manipulación de datos |
| `matplotlib` / `seaborn` | Visualización |
| `mlflow` | Tracking de experimentos |
| `joblib` | Serialización de modelos |

## 🧪 Tests

```bash
# Ejecutar tests
pytest tests/
```

## 📝 Tracking de Experimentos

El proyecto utiliza **MLflow** para el tracking de experimentos: 

```bash
# Iniciar UI de MLflow
mlflow ui
```

Accede a `http://localhost:5000` para visualizar los experimentos.

## 👥 Equipo 3

Proyecto desarrollado como parte del Bootcamp de Inteligencia Artificial - Proyecto 5 (NLP).

## 📄 Licencia

Este proyecto es parte de un bootcamp educativo. 

---

⭐ **¿Te ha sido útil este proyecto? ** ¡Dale una estrella al repositorio! 
