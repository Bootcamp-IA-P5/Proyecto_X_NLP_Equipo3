"""
🛡️ Demo del Sistema de Detección de Mensajes de Odio
==================================================

Este script demuestra cómo usar el modelo Random Forest entrenado
para detectar automáticamente mensajes de odio en comentarios.

Uso:
    python demo_hate_detection.py
"""

import joblib
import pandas as pd
from pathlib import Path
import re
import warnings

warnings.filterwarnings("ignore")

# Configurar rutas
project_root = Path(__file__).parent
models_dir = project_root / "models"


def load_model_components():
    """
    Carga el modelo entrenado y sus componentes
    """
    print("📦 Cargando modelo y componentes...")

    try:
        model = joblib.load(models_dir / "random_forest_hate_detector.pkl")
        tfidf = joblib.load(models_dir / "tfidf_vectorizer.pkl")
        scaler = joblib.load(models_dir / "feature_scaler.pkl")
        metadata = joblib.load(models_dir / "model_metadata.pkl")

        print(
            f"✅ Modelo cargado: {metadata['model_type']} ({metadata['model_version']})"
        )
        print(f"   F1-Score: {metadata['performance']['test_f1']:.4f}")
        print(f"   Recall: {metadata['performance']['test_recall']:.4f}")

        return model, tfidf, scaler, metadata

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(
            "💡 Ejecuta primero el notebook de entrenamiento: models/random_forest_hate_detection.ipynb"
        )
        return None, None, None, None


def preprocess_text(text):
    """
    Preprocesa un comentario aplicando el mismo pipeline que en entrenamiento
    """
    # Importar aquí para evitar errores si NLTK no está configurado
    try:
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
    except ImportError:
        print("❌ Error: NLTK no está instalado o configurado")
        return text.lower()

    # Limpieza básica
    text = str(text)
    text = re.sub(r"http[s]?://[^\s]+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)  # Menciones
    text = re.sub(r"#(\w+)", r"\1", text)  # Hashtags
    text = re.sub(r"\n|\t", " ", text)  # Saltos de línea
    text = re.sub(r"\s+", " ", text)  # Espacios múltiples

    # Normalización
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Números
    text = text.strip()

    # Stopwords y lemmatización
    try:
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        words = text.split()
        words = [word for word in words if word.lower() not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]

        return " ".join(words)
    except:
        # Fallback si NLTK no está configurado
        return text


def extract_features_single(original_text, processed_text):
    """
    Extrae features numéricas de un comentario individual
    """
    features = {}

    # Longitudes
    features["text_length"] = len(processed_text)
    features["word_count"] = len(processed_text.split())
    features["avg_word_length"] = features["text_length"] / max(
        features["word_count"], 1
    )

    # Características de puntuación (del texto original)
    features["exclamation_count"] = original_text.count("!")
    features["question_count"] = original_text.count("?")
    features["uppercase_ratio"] = sum(1 for c in original_text if c.isupper()) / max(
        len(original_text), 1
    )

    # Características del texto procesado
    words = processed_text.split()
    features["unique_word_ratio"] = len(set(words)) / max(len(words), 1) if words else 0

    return list(features.values())


def predict_hate_comment(text, model, vectorizer, scaler, show_details=False):
    """
    Predice si un comentario es de odio
    """
    from scipy import sparse

    # Preprocesar
    processed_text = preprocess_text(text)

    # Vectorización TF-IDF
    text_vector = vectorizer.transform([processed_text])

    # Features numéricas
    numeric_features = extract_features_single(text, processed_text)
    numeric_features_scaled = scaler.transform([numeric_features])

    # Combinar features
    combined_features = sparse.hstack([text_vector, numeric_features_scaled])

    # Predicción
    prediction = model.predict(combined_features)[0]
    probability = model.predict_proba(combined_features)[0, 1]

    # Resultado
    result = {
        "is_hate": bool(prediction),
        "hate_probability": float(probability),
        "confidence": "HIGH" if probability > 0.7 or probability < 0.3 else "MEDIUM",
        "processed_text": processed_text,
    }

    if show_details:
        feature_names = [
            "text_length",
            "word_count",
            "avg_word_length",
            "exclamation_count",
            "question_count",
            "uppercase_ratio",
            "unique_word_ratio",
        ]
        result["features"] = dict(zip(feature_names, numeric_features))

    return result


def demo_interactive():
    """
    Demo interactivo para probar el modelo
    """
    print("\n" + "=" * 80)
    print("🛡️  DEMO INTERACTIVO - DETECCIÓN DE MENSAJES DE ODIO")
    print("=" * 80)

    # Cargar modelo
    model, tfidf, scaler, metadata = load_model_components()
    if model is None:
        return

    print(f"\n💡 Ingresa comentarios para analizar (escribe 'quit' para salir)")
    print(
        f"   El modelo fue entrenado con {metadata['training_data']['n_samples']} comentarios de YouTube"
    )

    while True:
        try:
            comment = input("\n🗨️  Comentario: ").strip()

            if comment.lower() in ["quit", "exit", "q"]:
                print("👋 ¡Hasta luego!")
                break

            if not comment:
                continue

            # Predecir
            result = predict_hate_comment(
                comment, model, tfidf, scaler, show_details=True
            )

            # Mostrar resultado
            print(f"\n📊 RESULTADO:")

            if result["is_hate"]:
                print(f"   🚨 MENSAJE DE ODIO detectado")
                print(f"   🎯 Probabilidad: {result['hate_probability']:.1%}")
            else:
                print(f"   ✅ Mensaje NORMAL")
                print(f"   🎯 Probabilidad de odio: {result['hate_probability']:.1%}")

            print(f"   🔍 Confianza: {result['confidence']}")
            print(f"   📝 Texto procesado: '{result['processed_text']}'")

            # Mostrar features si es interesante
            if (
                result["features"]["exclamation_count"] > 0
                or result["features"]["uppercase_ratio"] > 0.1
            ):
                print(f"   📈 Features relevantes:")
                if result["features"]["exclamation_count"] > 0:
                    print(
                        f"      - Signos de exclamación: {result['features']['exclamation_count']}"
                    )
                if result["features"]["uppercase_ratio"] > 0.1:
                    print(
                        f"      - Ratio de mayúsculas: {result['features']['uppercase_ratio']:.1%}"
                    )

        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def demo_examples():
    """
    Demo con ejemplos predefinidos
    """
    print("\n" + "=" * 80)
    print("🧪 DEMO CON EJEMPLOS PREDEFINIDOS")
    print("=" * 80)

    # Cargar modelo
    model, tfidf, scaler, metadata = load_model_components()
    if model is None:
        return

    # Ejemplos de comentarios
    test_comments = [
        "This is a great video, thanks for sharing!",
        "You are such an idiot, go kill yourself!",
        "I disagree with your opinion but I respect your point of view",
        "All Muslims are terrorists and should be banned",
        "WHAT THE HELL IS WRONG WITH YOU?!?!",
        "Amazing content, keep it up!",
        "This channel is garbage and so are you",
        "Can you please explain this concept again?",
    ]

    print(f"\n🔍 Analizando {len(test_comments)} comentarios de ejemplo:\n")
    print(f"{'Comentario':<50} {'Predicción':<12} {'Probabilidad':<12} {'Confianza'}")
    print("-" * 85)

    for comment in test_comments:
        result = predict_hate_comment(comment, model, tfidf, scaler)

        # Formatear resultado
        label = "🚨 ODIO" if result["is_hate"] else "✅ Normal"
        prob = f"{result['hate_probability']:.1%}"
        confidence = result["confidence"]

        # Truncar comentario si es muy largo
        display_comment = comment[:45] + "..." if len(comment) > 45 else comment

        print(f"{display_comment:<50} {label:<12} {prob:<12} {confidence}")


def main():
    """
    Función principal del demo
    """
    print("🛡️ Sistema de Detección de Mensajes de Odio en YouTube")
    print("=" * 60)
    print("💡 Este sistema usa Random Forest para detectar automáticamente")
    print("   mensajes de odio en comentarios de YouTube.")
    print()
    print("📋 Opciones:")
    print("   1. Demo con ejemplos predefinidos")
    print("   2. Demo interactivo (ingresa tus propios comentarios)")

    while True:
        try:
            choice = input(
                "\n🎯 Selecciona una opción (1/2) o 'q' para salir: "
            ).strip()

            if choice in ["q", "quit", "exit"]:
                print("👋 ¡Hasta luego!")
                break
            elif choice == "1":
                demo_examples()
            elif choice == "2":
                demo_interactive()
            else:
                print("❌ Opción inválida. Usa 1, 2 o 'q'")

        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break


if __name__ == "__main__":
    main()
