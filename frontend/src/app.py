import streamlit as st
import pandas as pd
import pickle
import os
from pathlib import Path
import numpy as np
import joblib
import glob
import torch 

# Configuración de la página
st.set_page_config(
    page_title="Detección de Mensajes de Odio",
    page_icon="🛡️",
    layout="wide"
)

MODELS_DIR = "models"
DATASET_PATH = r"C:\Users\Administrator\Desktop\NLP\Proyecto_X_NLP_Equipo3\data\processed\youtube_all_versions.pkl"

# ==================== CLASE DETECTOR BERT ====================

class HateSpeechDetector:
    """
    Detector de discurso de odio para modelos BERT guardados con model_state_dict. 
    """
    
    def __init__(self, model_path):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
        
        with open(model_path, "rb") as f:
            self.package = pickle.load(f)
        
        self.device = torch.device("cuda" if torch. cuda.is_available() else "cpu")
        self.max_length = self.package.get("max_length", 128)
        self.threshold = self.package. get("optimal_threshold", 0.5)
        
        model_name = self. package.get("model_name", "bert-base-multilingual-cased")
        self.tokenizer = AutoTokenizer. from_pretrained(model_name)
        
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = self.package. get("num_labels", 2)
        
        if "config" in self. package:
            config.hidden_dropout_prob = self.package["config"].get("hidden_dropout_prob", 0.1)
            config. attention_probs_dropout_prob = self.package["config"]. get("attention_probs_dropout_prob", 0.1)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        self.model.load_state_dict(self.package["model_state_dict"])
        self.model.to(self. device)
        self.model.eval()
    
    def predict(self, text):
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self. device)
            
            outputs = self. model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch. softmax(outputs. logits, dim=1)
            prob_hate = probs[0, 1].item()
            
            is_hate = int(prob_hate >= self.threshold)
            return is_hate, np.array([1 - prob_hate, prob_hate])
        
# ==================== CARGA DE MODELOS ====================

@st.cache_resource
def load_models_from_folder(models_folder=MODELS_DIR):
    """
    Carga automáticamente todos los modelos desde la carpeta especificada.
    Soporta: 
      - Diccionarios con 'model_state_dict' (BERT guardado con pesos)
      - Diccionarios 'model_artifacts' (XGBoost + vectorizer + scalers + threshold)
      - Diccionarios 'random_forest_artifacts' (Random Forest + vectorizer + scaler)
      - Paquetes {'model':  .. ., 'tokenizer': ...} (BERT)
      - Modelos sklearn / xgboost sueltos
    """
    models = {}
    models_path = Path(models_folder)

    if not models_path. exists():
        return models

    pattern = str(models_path / "*.pkl")
    for path in glob.glob(pattern):
        file_path = Path(path)
        filename = file_path. name
        model_name = file_path.stem

        try:
            try:
                obj = joblib.load(file_path)
            except Exception: 
                with open(file_path, "rb") as f:
                    obj = pickle.load(f)

            entry = {
                "model":  obj,
                "path": str(file_path),
                "type": None,
            }

            # Detectar modelo BERT con model_state_dict (tu modelo)
            if isinstance(obj, dict) and "model_state_dict" in obj and "model_name" in obj:
                entry["type"] = "bert_state_dict"
                try: 
                    detector = HateSpeechDetector(str(file_path))
                    entry["detector"] = detector
                    entry["metrics"] = obj. get("metrics", {})
                    entry["threshold"] = obj.get("optimal_threshold", 0.5)
                except Exception as e: 
                    st. warning(f"Error cargando detector BERT {model_name}: {e}")
                models[model_name] = entry
                continue

            if isinstance(obj, dict) and "model" in obj and "vectorizer" in obj: 
                # Detectar si es Random Forest o XGBoost
                inner_model = obj["model"]
                if hasattr(inner_model, '__class__') and 'RandomForest' in inner_model.__class__.__name__: 
                    entry["type"] = "random_forest_artifacts"
                else:
                    entry["type"] = "xgboost_artifacts"
            elif isinstance(obj, dict) and "model" in obj and "tokenizer" in obj: 
                entry["type"] = "bert_package"
            elif hasattr(obj, "predict"):
                entry["type"] = "sklearn_or_xgb"
            else: 
                entry["type"] = "unknown"

            models[model_name] = entry

        except Exception as e: 
            st.warning(f"Error cargando {filename}: {e}")
            continue

    return models


# ==================== CARGA DEL DATASET ====================

@st.cache_data
def load_dataset(dataset_path):
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception:
        return None


# ==================== PREDICCIÓN ====================

def predict_with_xgboost_artifacts(text, artifacts):
    from scipy.sparse import hstack, csr_matrix
    import re

    raw_model = artifacts["model"]
    vectorizer = artifacts["vectorizer"]
    scaler = artifacts["scaler"]
    scaler_advanced = artifacts.get("scaler_advanced")
    threshold = artifacts.get("threshold", 0.5)
    feature_columns = artifacts.get("feature_columns", [])

    calibrator = None
    model = raw_model
    if isinstance(raw_model, tuple):
        model = raw_model[0]
        if len(raw_model) > 1:
            calibrator = raw_model[1]
        if len(raw_model) > 2 and artifacts.get("threshold") is None:
            try:
                threshold = float(raw_model[2])
            except Exception:
                pass

    text_tfidf = vectorizer.transform([text])

    if feature_columns:
        num_features = np.zeros((1, len(feature_columns)))
        num_features_scaled = scaler.transform(num_features)
    else:
        num_features_scaled = np.zeros((1, 0))

    def extract_advanced_features(text):
        offensive_words = [
            'hate', 'stupid', 'idiot', 'dumb', 'kill', 'die', 'death',
            'ugly', 'worst', 'terrible', 'awful', 'disgusting', 'pathetic',
            'loser', 'trash', 'garbage', 'shit', 'fuck', 'damn',
        ]
        text_lower = text.lower()
        words = text_lower.split()

        features = {}
        features['offensive_word_count'] = sum(1 for w in words if any(off in w for off in offensive_words))
        features['offensive_word_ratio'] = features['offensive_word_count'] / len(words) if words else 0
        features['avg_word_len'] = np.mean([len(w) for w in words]) if words else 0
        features['max_word_len'] = max([len(w) for w in words]) if words else 0
        features['char_repetition'] = len(re.findall(r'(.)\1{2,}', text))
        features['caps_words'] = sum(1 for w in words if w.isupper() and len(w) > 1)
        features['multiple_punctuation'] = len(re.findall(r'[!?]{2,}', text))
        negations = ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', "n't"]
        features['negation_count'] = sum(1 for w in words if w in negations)
        pronouns = ['you', 'your', 'they', 'them', 'their', 'he', 'she', 'his', 'her']
        features['pronoun_count'] = sum(1 for w in words if w in pronouns)
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0

        return features

    adv = extract_advanced_features(text)
    advanced_array = np.array([[
        adv['offensive_word_count'],
        adv['offensive_word_ratio'],
        adv['avg_word_len'],
        adv['max_word_len'],
        adv['char_repetition'],
        adv['caps_words'],
        adv['multiple_punctuation'],
        adv['negation_count'],
        adv['pronoun_count'],
        adv['unique_word_ratio'],
    ]])

    if scaler_advanced is not None:
        advanced_scaled = scaler_advanced.transform(advanced_array)
    else:
        advanced_scaled = advanced_array

    X_combined = hstack([
        text_tfidf,
        csr_matrix(num_features_scaled),
        csr_matrix(advanced_scaled)
    ])

    if calibrator is not None:
        base_proba = model.predict_proba(X_combined)[:, 1].reshape(-1, 1)
        proba_1 = calibrator.predict_proba(base_proba)[:, 1]
        proba = np.stack([1 - proba_1, proba_1], axis=1)[0]
    else:
        proba = model.predict_proba(X_combined)[0]

    pred = int(proba[1] >= threshold)
    return pred, proba

def predict_with_random_forest(text, artifacts):
    """
    Predicción usando Random Forest con TF-IDF y features numéricas básicas.
    """
    from scipy. sparse import hstack, csr_matrix

    model = artifacts["model"]
    vectorizer = artifacts["vectorizer"]
    scaler = artifacts["scaler"]
    threshold = artifacts. get("threshold", 0.5)
    feature_columns = artifacts. get("feature_columns", [])

    # Vectorizar texto
    text_tfidf = vectorizer.transform([text])

    # Features numéricas (inicializadas en cero si no hay texto procesado)
    if feature_columns:
        num_features = np.zeros((1, len(feature_columns)))
        num_features_scaled = scaler.transform(num_features)
    else:
        num_features_scaled = np.zeros((1, 0))

    # Combinar features (SIN features avanzadas)
    X_combined = hstack([
        text_tfidf,
        csr_matrix(num_features_scaled)
    ])

    # Predicción
    proba = model. predict_proba(X_combined)[0]
    pred = int(proba[1] >= threshold)
    
    return pred, proba

def predict_text(text, model_name, models):
    try:
        model_entry = models[model_name]
        model_data = model_entry["model"]
        model_type = model_entry.get("type", "unknown")

        # Manejar modelo BERT con state_dict (tu modelo)
        if model_type == "bert_state_dict": 
            detector = model_entry.get("detector")
            if detector: 
                return detector. predict(text)
            else:
                st.error("Detector BERT no inicializado")
                return None, None

        if model_type == "xgboost_artifacts":
            return predict_with_xgboost_artifacts(text, model_data)

        if model_type == "random_forest_artifacts":
            return predict_with_random_forest(text, model_data)

        if model_type == "bert_package" or (isinstance(model_data, dict) and 'model' in model_data and 'tokenizer' in model_data):
            import torch

            tokenizer = model_data['tokenizer']
            model = model_data['model']
            max_len = model_data. get('max_len', 128)

            model.eval()
            device = next(model.parameters()).device

            inputs = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_len,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                prediction = int(torch.argmax(logits, dim=1).cpu().item())

            return prediction, probs

        if hasattr(model_data, 'config') and hasattr(model_data.config, '_name_or_path'):
            import torch
            from transformers import AutoTokenizer

            model = model_data
            model_name_hf = model. config._name_or_path
            tokenizer = AutoTokenizer.from_pretrained(model_name_hf)

            model.eval()
            device = next(model.parameters()).device

            inputs = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            inputs = {k:  v.to(device) for k, v in inputs.items()}

            with torch. no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                prediction = int(torch.argmax(logits, dim=1).cpu().item())

            return prediction, probs

        if hasattr(model_data, 'predict'):
            prediction = model_data.predict([text])
            try:
                proba = model_data.predict_proba([text])
                return prediction[0], proba[0]
            except Exception:
                return prediction[0], None

        st.error(f"Tipo de modelo no reconocido: {type(model_data)} ({model_type})")
        return None, None

    except Exception as e: 
        st.error(f"Error en predicción: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

# ==================== CARGA GLOBAL ====================

st.title("🛡️ Sistema de Detección de Mensajes de Odio")

models = load_models_from_folder(MODELS_DIR)
dataset = load_dataset(DATASET_PATH) if os.path.exists(DATASET_PATH) else None

# Orden de pestañas: dataset, modelos, predicción, predicción en vídeo
tab_data, tab_models, tab_pred, tab_video = st.tabs([
    "📊 Información del Dataset",
    "📈 Análisis de Modelos",
    "🔍 Predicción en Vivo",
    "🎬 Predicción por Vídeo"
])


# ==================== PESTAÑA 1: Dataset ====================
with tab_data:
    st.header("Información del Dataset")

    if dataset is not None:
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        elif isinstance(dataset, dict):
            st.subheader("Versiones disponibles en el dataset")
            version_key = st.selectbox("Selecciona una versión:", list(dataset.keys()), key="data_version")
            df = dataset[version_key]
        else:
            st.write(f"Tipo de datos: {type(dataset)}")
            df = pd.DataFrame(dataset)

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

        st.subheader("Distribución de clases")
        if 'label' in df.columns or 'hate' in df.columns:
            label_col = 'label' if 'label' in df.columns else 'hate'
            dist = df[label_col].value_counts()
            st.bar_chart(dist)

        st.subheader("Muestra del dataset")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Información del dataset")
        st.write(df.describe())
    else:
        st.error(f"No se encontró o no se pudo cargar el archivo: {DATASET_PATH}")


# ==================== PESTAÑA 2: Modelos ====================
with tab_models:
    st.header("Análisis de Modelos Cargados")

    if models:
        st.subheader(f"Modelos disponibles: {len(models)}")

        models_info = []
        for name, info in models.items():
            model_type = info.get("type", type(info['model']).__name__)
            models_info.append({
                'Nombre': name,
                'Tipo': model_type,
                'Ruta': info['path']
            })

        models_df = pd.DataFrame(models_info)
        st.dataframe(models_df, use_container_width=True)

        st.subheader("Detalles del modelo")
        model_names = list(models.keys())
        selected_model_detail = st.selectbox("Selecciona un modelo para ver detalles:", model_names, key="detail")

        model_obj = models[selected_model_detail]['model']

        with st.expander("Ver atributos del modelo"):
            st.write(dir(model_obj))

        try:
            if hasattr(model_obj, 'get_params'):
                st.subheader("Parámetros del modelo")
                params = model_obj.get_params()
                st.json(params)
        except Exception:
            pass

        if dataset is not None:
            st.subheader("Prueba rápida de modelos")
            test_text = st.text_input("Texto de prueba:", "This is a test message")
            if st.button("Probar todos los modelos con texto de ejemplo"):
                if test_text:
                    results = []
                    for model_name in model_names:
                        pred, prob = predict_text(test_text, model_name, models)
                        results.append({
                            'Modelo': model_name,
                            'Predicción': 'Hate' if pred == 1 else 'Normal',
                            'Confianza': f"{max(prob) * 100:.1f}%" if prob is not None else "N/A"
                        })
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
    else:
        st.info("No hay modelos cargados")


# ==================== PESTAÑA 3: Predicción ====================
with tab_pred:
    st.header("Predicción de Mensajes en Vivo")

    if models:
        model_names = list(models.keys())
        selected_model = st.selectbox("Selecciona un modelo:", model_names, key="main_model_select")

        st.info(f"📁 Ruta del modelo: `{models[selected_model]['path']}`")
        st.write(f"Tipo detectado: {models[selected_model].get('type', 'desconocido')}")

        user_input = st.text_area(
            "Introduce el texto a analizar:",
            height=150,
            placeholder="Escribe aquí el mensaje que quieres analizar..."
        )

        if st.button("🔍 Analizar Mensaje", type="primary"):
            if user_input.strip():
                with st.spinner("Analizando..."):
                    prediction, probabilities = predict_text(user_input, selected_model, models)

                    if prediction is not None:
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            if prediction == 1:
                                st.error("⚠️ **HATE MESSAGE DETECTED**")
                            else:
                                st.success("✅ **Normal message (no hate)**")

                        with col2:
                            if probabilities is not None:
                                st.metric("Confidence", f"{max(probabilities) * 100:.1f}%")

                        if probabilities is not None:
                            st.subheader("Class probabilities")
                            prob_df = pd.DataFrame({
                                'Class': ['No hate', 'Hate'],
                                'Probability': probabilities
                            })
                            st.bar_chart(prob_df.set_index('Class'))
            else:
                st.warning("Please enter a message to analyze")

        with st.expander("📝 Example messages to try"):
            st.markdown("""
            **Normal messages:**
            - "I really like this video, very informative."
            - "Thank you for sharing this content."
            
            **Potential hate messages:**
            - "You are such an idiot, you know nothing."
            - "Get out of here, nobody wants you."
            """)
    else:
        st.warning("No models loaded. Please add models to the 'models' folder.")


# ==================== PESTAÑA 4: YouTube ====================
with tab_video:
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


# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/machine-learning.png", width=80)
    st.title("Panel")

    if models:
        model_names = list(models.keys())
        sidebar_model = st.selectbox("Modelo activo", model_names, key="sidebar_model_select")
        st.write(f"Modelo seleccionado: {sidebar_model}")
        st.success(f"✅ {len(models)} modelo(s) cargado(s)")
    else:
        st.error("❌ No hay modelos cargados")

    if dataset is not None:
        st.success("✅ Dataset cargado correctamente")
    else:
        st.error("❌ Dataset no cargado")
