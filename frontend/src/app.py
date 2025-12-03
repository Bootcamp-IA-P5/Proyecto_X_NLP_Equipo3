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
def load_models_from_folder():
    """Carga automáticamente todos los modelos desde la carpeta especificada"""
    models = {}
    
    # SOLUCIÓN: Detectar correctamente la raíz del proyecto
    # Desde frontend/src, subimos 2 niveles para llegar a la raíz
    current_file = Path(__file__) if '__file__' in globals() else None
    
    if current_file:
        # Si __file__ existe, usarlo
        script_dir = current_file.parent  # frontend/src
        project_root = script_dir.parent.parent  # raíz del proyecto
    else:
        # Si no, asumir que estamos en frontend/src
        script_dir = Path.cwd()
        # Si cwd está en frontend/src
        if script_dir.name == 'src' and script_dir.parent.name == 'frontend':
            project_root = script_dir.parent.parent
        # Si cwd está en frontend
        elif script_dir.name == 'frontend':
            project_root = script_dir.parent
        # Si cwd está en la raíz
        else:
            project_root = script_dir
    
    models_path = project_root / 'models'
    
    # Debug: mostrar rutas
    st.sidebar.info(f"📂 Current dir: `{Path.cwd()}`")
    st.sidebar.info(f"📂 Script dir: `{script_dir if current_file else 'N/A'}`")
    st.sidebar.info(f"📂 Raíz proyecto: `{project_root}`")
    st.sidebar.info(f"🔍 Buscando modelos en: `{models_path.absolute()}`")
    
    # Listar contenido del directorio raíz para debug
    if project_root.exists():
        st.sidebar.info(f"📁 Contenido de raíz:")
        for item in sorted(project_root.iterdir()):
            icon = "📁" if item.is_dir() else "📄"
            st.sidebar.text(f"  {icon} {item.name}")
    
    if not models_path.exists():
        st.error(f"❌ La carpeta no existe: {models_path.absolute()}")
        return models
    
    # Buscar archivos de modelos
    model_files = list(models_path.glob('*.pkl')) + list(models_path.glob('*.joblib')) + \
                  list(models_path.glob('*.h5')) + list(models_path.glob('*.pt')) + \
                  list(models_path.glob('*.bin'))
    
    st.sidebar.success(f"✅ Carpeta encontrada\n📁 Archivos: {len(model_files)}")
    
    for file_path in model_files:
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            models[file_path.stem] = {
                'model': model,
                'path': str(file_path),
                'size': f"{file_path.stat().st_size / (1024*1024):.2f} MB"
            }
            st.sidebar.success(f"✅ {file_path.name}")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error en {file_path.name}: {str(e)[:30]}")
    
    return models

# Función para cargar el dataset
@st.cache_data
def load_dataset():
    """Carga el dataset desde archivo pickle"""
    try:
        # Usar la misma lógica para encontrar la raíz
        current_file = Path(__file__) if '__file__' in globals() else None
        
        if current_file:
            script_dir = current_file.parent
            project_root = script_dir.parent.parent
        else:
            script_dir = Path.cwd()
            if script_dir.name == 'src' and script_dir.parent.name == 'frontend':
                project_root = script_dir.parent.parent
            elif script_dir.name == 'frontend':
                project_root = script_dir.parent
            else:
                project_root = script_dir
        
        dataset_path = project_root / 'data' / 'processed' / 'youtube_all_versions.pkl'
        
        st.sidebar.info(f"📊 Dataset: `{dataset_path.absolute()}`")
        
        if not dataset_path.exists():
            st.error(f"❌ Dataset no encontrado: {dataset_path}")
            return None
        
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        st.sidebar.success("✅ Dataset cargado")
        return data
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        return None

# Función para hacer predicción
def predict_text(text, model_name, models):
    """Realiza predicción sobre un texto nuevo"""
    try:
        model = models[model_name]['model']
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
models = load_models_from_folder()

if not models:
    st.warning("⚠️ No se encontraron modelos")
    st.info("""
    **Asegúrate de ejecutar el comando desde la ubicación correcta:**
    
    Opción 1 (desde la raíz del proyecto):
    ```
    cd C:\\Users\\Administrator\\Desktop\\NLP\\Proyecto_X_NLP_Equipo3
    streamlit run frontend/src/app.py
    ```
    
    Opción 2 (desde frontend/src):
    ```
    cd C:\\Users\\Administrator\\Desktop\\NLP\\Proyecto_X_NLP_Equipo3\\frontend\\src
    streamlit run app.py
    ```
    """)

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["📊 Información del Dataset", "🔍 Predicción en Vivo", "📈 Análisis de Modelos"])

# ==================== PESTAÑA 1: Dataset ====================
with tab1:
    st.header("Información del Dataset")
    
    dataset = load_dataset()
    
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
        with st.expander("📈 Ver estadísticas detalladas"):
            st.write(df.describe())
            
            # Mostrar columnas disponibles
            st.subheader("Columnas disponibles")
            st.write(list(df.columns))

# ==================== PESTAÑA 2: Predicción en Vivo ====================
with tab2:
    st.header("Predicción de Mensajes en Vivo")
    
    if models:
        # Selector de modelo
        model_names = list(models.keys())
        selected_model = st.selectbox("Selecciona un modelo:", model_names)
        
        st.info(f"📁 **Modelo:** `{selected_model}`\n📦 **Tamaño:** {models[selected_model]['size']}")
        
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
        st.warning("No hay modelos cargados")

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
                'Tamaño': info['size']
            })
        
        models_df = pd.DataFrame(models_info)
        st.dataframe(models_df, use_container_width=True)
        
        # Selector para ver detalles de un modelo
        st.subheader("Detalles del modelo")
        selected_model_detail = st.selectbox("Selecciona un modelo:", model_names, key="detail")
        
        model_obj = models[selected_model_detail]['model']
        
        # Mostrar tipo de modelo
        st.write(f"**Tipo:** `{type(model_obj).__name__}`")
        
        # Mostrar atributos del modelo
        with st.expander("Ver atributos del modelo"):
            attrs = [attr for attr in dir(model_obj) if not attr.startswith('_')]
            st.code('\n'.join(attrs))
        
        # Intentar mostrar parámetros si es scikit-learn
        try:
            if hasattr(model_obj, 'get_params'):
                with st.expander("Ver parámetros del modelo"):
                    params = model_obj.get_params()
                    st.json(params)
        except:
            pass
        
        # Comparación de modelos
        st.subheader("Prueba rápida de modelos")
        test_text = st.text_input("Texto de prueba:", "Este es un mensaje de prueba")
        
        if st.button("🚀 Probar todos los modelos") and test_text:
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
    st.title("🛡️ Proyecto NLP")
    st.markdown("**Detección de Mensajes de Odio**")
    
    st.divider()
    
    if models:
        st.success(f"✅ {len(models)} modelo(s)")
        for name in models.keys():
            st.text(f"• {name}")
    else:
        st.error("❌ Sin modelos")
