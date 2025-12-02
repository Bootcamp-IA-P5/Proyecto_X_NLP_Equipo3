import streamlit as st
import pandas as pd
import pickle
import os
from pathlib import Path
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Detección de Mensajes de Odio",
    page_icon="🛡️",
    layout="wide"
)

# Función para cargar modelos automáticamente
@st.cache_resource
def load_models_from_folder(models_folder='models'):
    """Carga automáticamente todos los modelos desde la carpeta especificada"""
    models = {}
    models_path = Path(models_folder)
    
    if not models_path.exists():
        st.error(f"La carpeta {models_folder} no existe")
        return models
    
    for file_path in models_path.glob('*'):
        if file_path.suffix in ['.pkl', '.joblib', '.h5', '.pt', '.bin']:
            try:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                models[file_path.stem] = {
                    'model': model,
                    'path': str(file_path)
                }
            except Exception as e:
                st.warning(f"No se pudo cargar {file_path.name}: {e}")
    
    return models

# Función para cargar el dataset
@st.cache_data
def load_dataset(dataset_path):
    """Carga el dataset desde archivo pickle"""
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        return None

# Función para hacer predicción
def predict_text(text, model_name, models):
    """Realiza predicción sobre un texto nuevo"""
    try:
        model = models[model_name]['model']
        # Adapta esto según tu pipeline de modelos
        prediction = model.predict([text])
        
        # Si el modelo tiene predict_proba
        try:
            proba = model.predict_proba([text])
            return prediction[0], proba[0]
        except:
            return prediction[0], None
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return None, None

# Título principal
st.title("🛡️ Sistema de Detección de Mensajes de Odio")

# Cargar modelos
models = load_models_from_folder('models')

if not models:
    st.warning("No se encontraron modelos en la carpeta 'models'")

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["📊 Información del Dataset", "🔍 Predicción en Vivo", "📈 Análisis de Modelos"])

# ==================== PESTAÑA 1: Dataset ====================
with tab1:
    st.header("Información del Dataset")
    
    dataset_path = r"C:\Users\Administrator\Desktop\NLP\Proyecto_X_NLP_Equipo3\data\processed\youtube_all_versions.pkl"
    
    if os.path.exists(dataset_path):
        dataset = load_dataset(dataset_path)
        
        if dataset is not None:
            # Si es DataFrame
            if isinstance(dataset, pd.DataFrame):
                df = dataset
            # Si es diccionario con DataFrames
            elif isinstance(dataset, dict):
                st.subheader("Versiones disponibles en el dataset")
                version_key = st.selectbox("Selecciona una versión:", list(dataset.keys()))
                df = dataset[version_key]
            else:
                st.write(f"Tipo de datos: {type(dataset)}")
                df = pd.DataFrame(dataset)
            
            # Mostrar información básica
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de muestras", len(df))
            with col2:
                if 'label' in df.columns or 'hate' in df.columns:
                    label_col = 'label' if 'label' in df.columns else 'hate'
                    hate_count = df[label_col].sum() if df[label_col].dtype in ['int64', 'float64'] else len(df[df[label_col] == 1])
                    st.metric("Mensajes de odio", hate_count)
            with col3:
                if 'label' in df.columns or 'hate' in df.columns:
                    label_col = 'label' if 'label' in df.columns else 'hate'
                    hate_pct = (hate_count / len(df)) * 100
                    st.metric("% Odio", f"{hate_pct:.1f}%")
            
            # Mostrar distribución
            st.subheader("Distribución de clases")
            if 'label' in df.columns or 'hate' in df.columns:
                label_col = 'label' if 'label' in df.columns else 'hate'
                dist = df[label_col].value_counts()
                st.bar_chart(dist)
            
            # Mostrar muestra del dataset
            st.subheader("Muestra del dataset")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Estadísticas adicionales
            st.subheader("Información del dataset")
            st.write(df.describe())
    else:
        st.error(f"No se encontró el archivo: {dataset_path}")

# ==================== PESTAÑA 2: Predicción en Vivo ====================
with tab2:
    st.header("Predicción de Mensajes en Vivo")
    
    if models:
        # Selector de modelo
        model_names = list(models.keys())
        selected_model = st.selectbox("Selecciona un modelo:", model_names)
        
        st.info(f"📁 Ruta del modelo: `{models[selected_model]['path']}`")
        
        # Área de texto para input
        user_input = st.text_area(
            "Introduce el texto a analizar:",
            height=150,
            placeholder="Escribe aquí el mensaje que quieres analizar..."
        )
        
        # Botón de predicción
        if st.button("🔍 Analizar Mensaje", type="primary"):
            if user_input.strip():
                with st.spinner("Analizando..."):
                    prediction, probabilities = predict_text(user_input, selected_model, models)
                    
                    if prediction is not None:
                        # Mostrar resultado
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if prediction == 1:
                                st.error("⚠️ **MENSAJE DE ODIO DETECTADO**")
                            else:
                                st.success("✅ **Mensaje normal (sin odio)**")
                        
                        with col2:
                            if probabilities is not None:
                                st.metric("Confianza", f"{max(probabilities)*100:.1f}%")
                        
                        # Mostrar probabilidades si están disponibles
                        if probabilities is not None:
                            st.subheader("Probabilidades por clase")
                            prob_df = pd.DataFrame({
                                'Clase': ['Sin odio', 'Odio'],
                                'Probabilidad': probabilities
                            })
                            st.bar_chart(prob_df.set_index('Clase'))
            else:
                st.warning("Por favor, introduce un texto para analizar")
        
        # Ejemplos de prueba
        with st.expander("📝 Ejemplos de mensajes para probar"):
            st.markdown("""
            **Ejemplos de mensajes normales:**
            - "Me encanta este video, muy informativo"
            - "Gracias por compartir este contenido"
            
            **Ejemplos de posibles mensajes de odio:**
            - "Qué idiota eres, no sabes nada"
            - "Vete de aquí, no te queremos"
            """)
    else:
        st.warning("No hay modelos cargados. Asegúrate de tener modelos en la carpeta 'models'")

# ==================== PESTAÑA 3: Análisis de Modelos ====================
with tab3:
    st.header("Análisis de Modelos Cargados")
    
    if models:
        st.subheader(f"Modelos disponibles: {len(models)}")
        
        # Tabla con información de modelos
        models_info = []
        for name, info in models.items():
            model_type = type(info['model']).__name__
            models_info.append({
                'Nombre': name,
                'Tipo': model_type,
                'Ruta': info['path']
            })
        
        models_df = pd.DataFrame(models_info)
        st.dataframe(models_df, use_container_width=True)
        
        # Selector para ver detalles de un modelo
        st.subheader("Detalles del modelo")
        selected_model_detail = st.selectbox("Selecciona un modelo para ver detalles:", model_names, key="detail")
        
        model_obj = models[selected_model_detail]['model']
        
        # Mostrar atributos del modelo
        with st.expander("Ver atributos del modelo"):
            st.write(dir(model_obj))
        
        # Intentar mostrar parámetros si es scikit-learn
        try:
            if hasattr(model_obj, 'get_params'):
                st.subheader("Parámetros del modelo")
                params = model_obj.get_params()
                st.json(params)
        except:
            pass
        
        # Comparación de modelos (si hay dataset cargado)
        if os.path.exists(dataset_path):
            st.subheader("Prueba rápida de modelos")
            if st.button("Probar todos los modelos con texto de ejemplo"):
                test_text = st.text_input("Texto de prueba:", "Este es un mensaje de prueba")
                
                if test_text:
                    results = []
                    for model_name in model_names:
                        pred, prob = predict_text(test_text, model_name, models)
                        results.append({
                            'Modelo': model_name,
                            'Predicción': 'Odio' if pred == 1 else 'Normal',
                            'Confianza': f"{max(prob)*100:.1f}%" if prob is not None else "N/A"
                        })
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
    else:
        st.info("No hay modelos cargados")

# Sidebar con información adicional
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/machine-learning.png", width=80)
    st.title("Información")
    st.markdown("""
    ### 🎯 Proyecto NLP
    **Detección de Mensajes de Odio**
    
    #### 📂 Carpetas:
    - `models/`: Modelos ML
    - `data/processed/`: Datasets
    
    #### 🔧 Funcionalidades:
    - ✅ Carga automática de modelos
    - ✅ Predicción en tiempo real
    - ✅ Análisis del dataset
    - ✅ Comparación de modelos
    """)
    
    if models:
        st.success(f"✅ {len(models)} modelo(s) cargado(s)")
    else:
        st.error("❌ No hay modelos cargados")
