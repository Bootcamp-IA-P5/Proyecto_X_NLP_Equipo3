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

# ==================== FUNCIONES DE CARGA ====================

def get_project_root():
    """Obtiene la raíz del proyecto desde frontend/src"""
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
    
    return project_root

def load_single_model(file_path):
    """Intenta cargar un modelo con diferentes métodos"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        pass
    
    try:
        import joblib
        return joblib.load(file_path)
    except:
        pass
    
    try:
        import torch
        return torch.load(file_path, map_location='cpu')
    except:
        pass
    
    try:
        import cloudpickle
        with open(file_path, 'rb') as f:
            return cloudpickle.load(f)
    except:
        pass
    
    return None

@st.cache_resource
def load_models_from_folder():
    """Carga automáticamente todos los modelos desde la carpeta models"""
    models = {}
    errors = []
    project_root = get_project_root()
    models_path = project_root / 'models'
    
    if not models_path.exists():
        return models, "❌ Carpeta models no encontrada", errors
    
    model_files = list(models_path.glob('*.pkl')) + list(models_path.glob('*.joblib')) + \
                  list(models_path.glob('*.h5')) + list(models_path.glob('*.pt')) + \
                  list(models_path.glob('*.bin'))
    
    for file_path in model_files:
        model = load_single_model(file_path)
        if model is not None:
            models[file_path.stem] = {
                'model': model,
                'path': str(file_path),
                'size': f"{file_path.stat().st_size / (1024*1024):.2f} MB"
            }
        else:
            errors.append(file_path.name)
    
    status = f"✅ {len(models)}/{len(model_files)} modelo(s)" if models else "❌ Sin modelos"
    return models, status, errors

@st.cache_data
def load_dataset():
    """Carga el dataset desde archivo pickle"""
    try:
        project_root = get_project_root()
        dataset_path = project_root / 'data' / 'processed' / 'youtube_all_versions.pkl'
        
        if not dataset_path.exists():
            return None, "❌ Dataset no encontrado"
        
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        return data, "✅ Dataset cargado"
    except Exception as e:
        return None, f"❌ Error: {str(e)[:30]}"

def predict_text(text, model_name, models):
    """Realiza predicción sobre un texto nuevo - compatible con sklearn y transformers"""
    try:
        model_data = models[model_name]['model']
        
        # Caso 1: Es un diccionario con modelo y tokenizer (típico de BERT guardado manualmente)
        if isinstance(model_data, dict):
            if 'model' in model_data and 'tokenizer' in model_data:
                import torch
                tokenizer = model_data['tokenizer']
                model = model_data['model']
                model.eval()
                
                inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                    prediction = int(torch.argmax(logits, dim=1).item())
                
                return prediction, probs
        
        # Caso 2: Es un pipeline de transformers
        if hasattr(model_data, 'task') or str(type(model_data).__name__) == 'TextClassificationPipeline':
            result = model_data(text)
            if isinstance(result, list):
                result = result[0]
            label = result.get('label', '')
            score = result.get('score', 0)
            
            # Convertir label a 0/1
            if 'LABEL_1' in label or 'hate' in label.lower() or 'positive' in label.lower():
                prediction = 1
                probs = [1 - score, score]
            else:
                prediction = 0
                probs = [score, 1 - score]
            
            return prediction, np.array(probs)
        
        # Caso 3: Modelo sklearn estándar con predict
        if hasattr(model_data, 'predict'):
            prediction = model_data.predict([text])
            
            try:
                proba = model_data.predict_proba([text])
                return prediction[0], proba[0]
            except:
                return prediction[0], None
        
        # Caso 4: Modelo PyTorch directo
        if hasattr(model_data, 'forward') or hasattr(model_data, 'eval'):
            st.warning(f"El modelo {model_name} parece ser PyTorch pero necesita un tokenizer. Verifica cómo fue guardado.")
            return None, None
        
        st.error(f"No se reconoce el tipo de modelo: {type(model_data)}")
        return None, None
        
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return None, None

def extract_video_id(url):
    """Extrae el ID del video de una URL de YouTube"""
    import re
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_comments_api(video_id, api_key, max_comments=100):
    """Obtiene comentarios usando la API oficial de YouTube"""
    try:
        from googleapiclient.discovery import build
        
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        comments = []
        next_page_token = None
        
        while len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat='plainText'
            )
            response = request.execute()
            
            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'autor': snippet.get('authorDisplayName', 'Desconocido'),
                    'texto': snippet.get('textDisplay', ''),
                    'likes': snippet.get('likeCount', 0),
                    'fecha': snippet.get('publishedAt', '')
                })
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        
        return comments[:max_comments], None
    except ImportError:
        return None, "❌ Instala: pip install google-api-python-client"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ==================== CARGAR DATOS ====================

models, models_status, model_errors = load_models_from_folder()
dataset, dataset_status = load_dataset()

# ==================== SIDEBAR ====================

with st.sidebar:
    st.title("🛡️ Panel de Control")
    
    st.divider()
    
    st.subheader("📦 Estado")
    st.write(models_status)
    st.write(dataset_status)
    
    if model_errors:
        with st.expander("⚠️ Modelos no cargados"):
            for err in model_errors:
                st.text(f"• {err}")
    
    st.divider()
    
    if models:
        st.subheader("🤖 Modelo activo")
        model_names = list(models.keys())
        selected_model = st.selectbox(
            "Selecciona modelo:",
            model_names,
            label_visibility="collapsed"
        )
        st.caption(f"📦 {models[selected_model]['size']}")
        
        # Mostrar tipo de modelo
        model_type = type(models[selected_model]['model']).__name__
        st.caption(f"🔧 Tipo: {model_type}")
    else:
        selected_model = None
        st.warning("Sin modelos disponibles")

# ==================== TÍTULO PRINCIPAL ====================

st.title("🛡️ Sistema de Detección de Mensajes de Odio")

# ==================== PESTAÑAS ====================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset", 
    "📈 Modelos",
    "🔍 Predicción en Vivo", 
    "🎬 YouTube"
])

# ==================== PESTAÑA 1: Dataset ====================
with tab1:
    st.header("Información del Dataset")
    
    if dataset is not None:
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        elif isinstance(dataset, dict):
            st.subheader("Versiones disponibles")
            version_key = st.selectbox("Selecciona una versión:", list(dataset.keys()))
            df = dataset[version_key] if isinstance(dataset[version_key], pd.DataFrame) else pd.DataFrame(dataset[version_key])
        else:
            df = pd.DataFrame(dataset)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de muestras", len(df))
        
        label_col = None
        for col in ['label', 'hate', 'odio', 'target', 'class']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col:
            with col2:
                hate_count = int(df[label_col].sum())
                st.metric("Mensajes de odio", hate_count)
            with col3:
                hate_pct = (hate_count / len(df)) * 100
                st.metric("% Odio", f"{hate_pct:.1f}%")
            
            st.subheader("Distribución de clases")
            dist = df[label_col].value_counts()
            st.bar_chart(dist)
        
        st.subheader("Muestra del dataset")
        st.dataframe(df.head(20), use_container_width=True)
        
        with st.expander("📈 Ver estadísticas detalladas"):
            st.write(f"**Columnas:** {list(df.columns)}")
            st.write(df.describe())
    else:
        st.error("No se pudo cargar el dataset")

# ==================== PESTAÑA 2: Modelos ====================
with tab2:
    st.header("Modelos Cargados")
    
    if models:
        models_info = []
        for name, info in models.items():
            model_obj = info['model']
            if isinstance(model_obj, dict):
                model_type = "BERT (dict)"
            else:
                model_type = type(model_obj).__name__
            models_info.append({
                'Nombre': name,
                'Tipo': model_type,
                'Tamaño': info['size']
            })
        
        st.dataframe(pd.DataFrame(models_info), use_container_width=True)
        
        if selected_model:
            st.subheader(f"Detalles: {selected_model}")
            model_obj = models[selected_model]['model']
            
            if isinstance(model_obj, dict):
                st.write("**Tipo:** Diccionario con modelo BERT")
                st.write(f"**Claves:** {list(model_obj.keys())}")
            else:
                st.write(f"**Tipo:** `{type(model_obj).__name__}`")
                try:
                    if hasattr(model_obj, 'get_params'):
                        with st.expander("Ver parámetros"):
                            st.json(model_obj.get_params())
                except:
                    pass
        
        st.subheader("Prueba rápida")
        test_text = st.text_input("Texto de prueba:", "Este es un mensaje de prueba")
        
        if st.button("🚀 Probar todos los modelos"):
            results = []
            for model_name in models.keys():
                with st.spinner(f"Probando {model_name}..."):
                    pred, prob = predict_text(test_text, model_name, models)
                    if pred is not None:
                        results.append({
                            'Modelo': model_name,
                            'Predicción': '🔴 Odio' if pred == 1 else '🟢 Normal',
                            'Confianza': f"{max(prob)*100:.1f}%" if prob is not None else "N/A"
                        })
                    else:
                        results.append({
                            'Modelo': model_name,
                            'Predicción': '❓ Error',
                            'Confianza': "N/A"
                        })
            
            st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("No hay modelos cargados")

# ==================== PESTAÑA 3: Predicción en Vivo ====================
with tab3:
    st.header("Predicción de Mensajes en Vivo")
    
    if models and selected_model:
        st.info(f"🤖 Usando modelo: **{selected_model}**")
        
        st.caption("⚠️ Los modelos están entrenados en inglés. Introduce textos en inglés para mejores resultados.")
        
        user_input = st.text_area(
            "Introduce el texto a analizar:",
            height=150,
            placeholder="Write here the message you want to analyze...",
            key="prediction_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("🔍 Analizar", type="primary", use_container_width=True)
        
        if analyze_button:
            if user_input.strip():
                with st.spinner("Analizando..."):
                    prediction, probabilities = predict_text(user_input, selected_model, models)
                    
                    st.divider()
                    
                    if prediction is not None:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if prediction == 1:
                                st.error("⚠️ **MENSAJE DE ODIO DETECTADO**")
                            else:
                                st.success("✅ **Mensaje normal (sin odio)**")
                        
                        with col2:
                            if probabilities is not None:
                                st.metric("Confianza", f"{max(probabilities)*100:.1f}%")
                        
                        if probabilities is not None:
                            st.subheader("Probabilidades")
                            prob_df = pd.DataFrame({
                                'Clase': ['Sin odio', 'Odio'],
                                'Probabilidad': [float(probabilities[0]), float(probabilities[1])]
                            })
                            st.bar_chart(prob_df.set_index('Clase'))
                    else:
                        st.error("❌ No se pudo realizar la predicción. Revisa el formato del modelo.")
            else:
                st.warning("Por favor, introduce un texto")
        
        with st.expander("📝 Ejemplos de mensajes (en inglés)"):
            st.markdown("""
            **Normal messages:**
            - "I love this video, very informative"
            - "Thanks for sharing this content"
            - "Great explanation, keep it up!"
            - "This is really helpful, thank you"
            
            **Possible hate speech:**
            - "You are such an idiot"
            - "Get out of here, nobody wants you"
            - "I hate people like you"
            - "You're worthless and stupid"
            """)
    else:
        st.warning("No hay modelos disponibles")

# ==================== PESTAÑA 4: YouTube ====================
with tab4:
    st.header("🎬 Análisis de Comentarios de YouTube")
    
    st.warning("""
    ⚠️ **Importante:** Se usa la API oficial de YouTube (seguro y legal).
    Necesitas una API Key gratuita de Google.
    """)
    
    if not models or not selected_model:
        st.error("⚠️ Necesitas tener al menos un modelo cargado")
    else:
        st.success(f"🤖 Usando modelo: **{selected_model}**")
        
        st.subheader("🔑 Configuración")
        api_key = st.text_input(
            "YouTube API Key:",
            type="password",
            help="Obtén tu API Key gratis en: https://console.cloud.google.com/apis/credentials"
        )
        
        if not api_key:
            st.info("""
            **¿Cómo obtener una API Key gratuita?**
            1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
            2. Crea un proyecto nuevo
            3. Habilita "YouTube Data API v3"
            4. Crea credenciales → API Key
            """)
        
        st.divider()
        
        youtube_url = st.text_input(
            "URL del video de YouTube:",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            max_comments = st.slider("Máximo de comentarios:", 10, 200, 50)
        with col2:
            video_id = extract_video_id(youtube_url) if youtube_url else None
            if video_id:
                st.success(f"✅ Video ID: {video_id}")
        
        if st.button("📥 Extraer y Analizar", type="primary", disabled=not api_key):
            if not api_key:
                st.error("❌ Introduce tu YouTube API Key")
            elif youtube_url and video_id:
                with st.spinner(f"Extrayendo hasta {max_comments} comentarios..."):
                    comments, error = get_youtube_comments_api(video_id, api_key, max_comments)
                    
                    if error:
                        st.error(error)
                    elif comments:
                        st.success(f"✅ {len(comments)} comentarios extraídos")
                        
                        with st.spinner("Analizando comentarios..."):
                            results = []
                            hate_count = 0
                            
                            progress_bar = st.progress(0)
                            for i, comment in enumerate(comments):
                                pred, prob = predict_text(comment['texto'], selected_model, models)
                                
                                is_hate = pred == 1 if pred is not None else False
                                if is_hate:
                                    hate_count += 1
                                
                                confidence = max(prob)*100 if prob is not None else 0
                                
                                results.append({
                                    'Autor': comment['autor'],
                                    'Comentario': comment['texto'][:100] + '...' if len(comment['texto']) > 100 else comment['texto'],
                                    'Clasificación': '🔴 Odio' if is_hate else '🟢 Normal',
                                    'Confianza': f"{confidence:.1f}%",
                                    'Likes': comment['likes']
                                })
                                
                                progress_bar.progress((i + 1) / len(comments))
                        
                        st.subheader("📊 Resumen")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total", len(comments))
                        with col2:
                            st.metric("Odio detectado", hate_count)
                        with col3:
                            pct = (hate_count / len(comments)) * 100
                            st.metric("% Odio", f"{pct:.1f}%")
                        
                        st.subheader("📝 Resultados")
                        
                        filter_option = st.radio(
                            "Filtrar:",
                            ["Todos", "Solo odio", "Solo normales"],
                            horizontal=True
                        )
                        
                        results_df = pd.DataFrame(results)
                        
                        if filter_option == "Solo odio":
                            results_df = results_df[results_df['Clasificación'] == '🔴 Odio']
                        elif filter_option == "Solo normales":
                            results_df = results_df[results_df['Clasificación'] == '🟢 Normal']
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Descargar CSV",
                            csv,
                            f"analisis_{video_id}.csv",
                            "text/csv"
                        )
                    else:
                        st.warning("No se encontraron comentarios")
            else:
                st.warning("Introduce una URL válida de YouTube")
