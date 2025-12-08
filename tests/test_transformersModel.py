"""
Tests para el modelo BERT Multilingual de clasificación de Hate Speech
Ejecutar con: pytest tests/test_transformers_model.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import pickle

# Agregar el directorio raíz al path para imports
ROOT_DIR = Path(__file__).parent. parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "models"))


# ============================================================================
# FUNCIÓN PARA CARGAR EL DETECTOR
# ============================================================================

def load_detector():
    """Carga el detector de hate speech."""
    model_path = ROOT_DIR / "models" / "hate_speech_model.pkl"
    
    if not model_path. exists():
        pytest.skip(f"Modelo no encontrado: {model_path}")
    
    # Importar aquí para evitar errores si el módulo no existe
    try:
        from hate_detector import HateSpeechDetector
        return HateSpeechDetector(str(model_path))
    except ImportError:
        # Si no puede importar, crear una versión inline
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
        
        class HateSpeechDetector:
            def __init__(self, model_path):
                with open(model_path, "rb") as f:
                    self.package = pickle.load(f)
                
                self.device = torch.device("cuda" if torch. cuda.is_available() else "cpu")
                self.max_length = self.package["max_length"]
                self.threshold = self.package["optimal_threshold"]
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.package["model_name"])
                
                config = AutoConfig.from_pretrained(self.package["model_name"])
                config.num_labels = self.package["num_labels"]
                config.hidden_dropout_prob = self.package["config"]["hidden_dropout_prob"]
                config.attention_probs_dropout_prob = self. package["config"]["attention_probs_dropout_prob"]
                
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.package["model_name"],
                    config=config,
                    ignore_mismatched_sizes=True
                )
                self.model.load_state_dict(self.package["model_state_dict"])
                self.model. to(self.device)
                self.model.eval()
            
            def predict(self, text):
                single_input = isinstance(text, str)
                texts = [text] if single_input else text
                results = []
                
                with torch.no_grad():
                    for t in texts:
                        encoding = self.tokenizer(
                            t,
                            max_length=self. max_length,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt"
                        )
                        
                        input_ids = encoding["input_ids"].to(self. device)
                        attention_mask = encoding["attention_mask"]. to(self.device)
                        
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        probs = torch.softmax(outputs.logits, dim=1)
                        prob_hate = probs[0, 1].item()
                        
                        is_hate = prob_hate >= self.threshold
                        
                        results.append({
                            "text": t,
                            "is_hate": is_hate,
                            "probability": prob_hate,
                            "label": "Hate" if is_hate else "No Hate",
                            "confidence": prob_hate if is_hate else 1 - prob_hate
                        })
                
                return results[0] if single_input else results
            
            def predict_batch(self, texts, batch_size=16):
                all_results = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    results = self.predict(batch)
                    if isinstance(results, dict):
                        results = [results]
                    all_results.extend(results)
                return all_results
            
            def get_metrics(self):
                return self.package["metrics"]
            
            def get_threshold(self):
                return self.threshold
        
        return HateSpeechDetector(str(model_path))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def detector():
    """Carga el detector una vez para todos los tests."""
    return load_detector()


@pytest.fixture
def sample_texts():
    """Textos de ejemplo para pruebas."""
    return [
        "This is a normal comment about the video",
        "I hate you and your stupid content",
        "Great video, thanks for sharing! ",
        "You are worthless and should disappear",
        "Love this channel, very informative"
    ]


@pytest.fixture
def sample_labels():
    """Labels correspondientes a los textos de ejemplo."""
    return [0, 1, 0, 1, 0]


@pytest.fixture
def model_path():
    """Ruta al modelo guardado."""
    return ROOT_DIR / "models" / "hate_speech_model.pkl"


# ============================================================================
# TESTS PARA CARGA DEL MODELO
# ============================================================================

class TestModelLoading:
    """Tests para la carga del modelo."""
    
    def test_model_file_exists(self, model_path):
        """Verifica que el archivo del modelo existe."""
        assert model_path.exists(), f"Modelo no encontrado: {model_path}"
    
    def test_model_loads_successfully(self, detector):
        """Verifica que el modelo se carga correctamente."""
        assert detector is not None
        assert detector.model is not None
        assert detector.tokenizer is not None
    
    def test_model_has_threshold(self, detector):
        """Verifica que el modelo tiene un umbral definido."""
        threshold = detector.get_threshold()
        assert threshold is not None
        assert 0 < threshold < 1
    
    def test_model_has_metrics(self, detector):
        """Verifica que el modelo tiene métricas guardadas."""
        metrics = detector.get_metrics()
        assert metrics is not None
        assert "test_f1_optimal" in metrics
        assert "test_precision" in metrics
        assert "test_recall" in metrics
    
    def test_model_package_has_required_keys(self, detector):
        """Verifica que el paquete del modelo tiene todas las claves necesarias."""
        required_keys = [
            'model_state_dict', 'model_name', 'num_labels',
            'max_length', 'config', 'optimal_threshold', 'metrics'
        ]
        
        for key in required_keys:
            assert key in detector. package, f"Falta clave: {key}"


# ============================================================================
# TESTS PARA PREDICCIÓN BÁSICA
# ============================================================================

class TestBasicPrediction:
    """Tests para predicción básica."""
    
    def test_predict_single_text(self, detector):
        """Verifica predicción con un solo texto."""
        result = detector. predict("This is a test message")
        
        assert result is not None
        assert "is_hate" in result
        assert "probability" in result
        assert "label" in result
        assert "confidence" in result
        assert "text" in result
    
    def test_predict_returns_correct_types(self, detector):
        """Verifica que los tipos de retorno son correctos."""
        result = detector.predict("Test message")
        
        assert isinstance(result["is_hate"], bool)
        assert isinstance(result["probability"], float)
        assert isinstance(result["label"], str)
        assert isinstance(result["confidence"], float)
    
    def test_probability_in_valid_range(self, detector):
        """Verifica que la probabilidad está entre 0 y 1."""
        result = detector.predict("Test message")
        
        assert 0 <= result["probability"] <= 1
        assert 0 <= result["confidence"] <= 1
    
    def test_label_is_valid(self, detector):
        """Verifica que la etiqueta es válida."""
        result = detector.predict("Test message")
        
        assert result["label"] in ["Hate", "No Hate"]
    
    def test_label_matches_is_hate(self, detector):
        """Verifica coherencia entre is_hate y label."""
        result = detector.predict("Test message")
        
        if result["is_hate"]:
            assert result["label"] == "Hate"
        else:
            assert result["label"] == "No Hate"


# ============================================================================
# TESTS PARA PREDICCIÓN POR LOTES
# ============================================================================

class TestBatchPrediction:
    """Tests para predicción por lotes."""
    
    def test_predict_multiple_texts(self, detector, sample_texts):
        """Verifica predicción con múltiples textos."""
        results = detector. predict(sample_texts)
        
        assert isinstance(results, list)
        assert len(results) == len(sample_texts)
    
    def test_predict_batch(self, detector):
        """Verifica predicción por lotes."""
        texts = ["Text " + str(i) for i in range(10)]
        results = detector. predict_batch(texts, batch_size=4)
        
        assert len(results) == 10
        for result in results:
            assert "is_hate" in result
            assert "probability" in result
    
    def test_batch_all_results_valid(self, detector, sample_texts):
        """Verifica que todos los resultados del batch son válidos."""
        results = detector. predict(sample_texts)
        
        for result in results:
            assert 0 <= result["probability"] <= 1
            assert result["label"] in ["Hate", "No Hate"]


# ============================================================================
# TESTS PARA DETECCIÓN DE ODIO
# ============================================================================

class TestHateDetection:
    """Tests para la detección de odio."""
    
    def test_clearly_hateful_text(self, detector):
        """Verifica que texto claramente ofensivo se detecta."""
        hateful_texts = [
            "I hate you and want you to die",
            "You are stupid and worthless",
            "Kill all of them"
        ]
        
        hate_count = 0
        for text in hateful_texts:
            result = detector.predict(text)
            if result["is_hate"]:
                hate_count += 1
        
        assert hate_count >= 2, f"Solo {hate_count}/3 textos de odio detectados"
    
    def test_clearly_neutral_text(self, detector):
        """Verifica que texto claramente neutral no se detecta como odio."""
        neutral_texts = [
            "I love this video, thanks for sharing",
            "Great content, very informative",
            "Have a wonderful day everyone"
        ]
        
        neutral_count = 0
        for text in neutral_texts:
            result = detector.predict(text)
            if not result["is_hate"]:
                neutral_count += 1
        
        assert neutral_count >= 2, f"Solo {neutral_count}/3 textos neutrales correctos"


# ============================================================================
# TESTS PARA MÉTRICAS DEL MODELO
# ============================================================================

class TestModelMetrics:
    """Tests para las métricas del modelo."""
    
    def test_model_f1_above_threshold(self, detector):
        """Verifica que el F1 guardado supera el mínimo esperado."""
        metrics = detector.get_metrics()
        
        assert metrics["test_f1_optimal"] >= 0.65, \
            f"F1 muy bajo: {metrics['test_f1_optimal']}"
    
    def test_model_precision_above_threshold(self, detector):
        """Verifica que la precisión supera el mínimo esperado."""
        metrics = detector.get_metrics()
        
        assert metrics["test_precision"] >= 0.60, \
            f"Precisión muy baja: {metrics['test_precision']}"
    
    def test_model_recall_above_threshold(self, detector):
        """Verifica que el recall supera el mínimo esperado."""
        metrics = detector. get_metrics()
        
        assert metrics["test_recall"] >= 0.60, \
            f"Recall muy bajo: {metrics['test_recall']}"
    
    def test_metrics_are_valid_values(self, detector):
        """Verifica que las métricas son valores válidos."""
        metrics = detector.get_metrics()
        
        for key in ["test_f1_optimal", "test_precision", "test_recall"]:
            assert 0 <= metrics[key] <= 1, f"Métrica {key} fuera de rango"


# ============================================================================
# TESTS PARA CASOS LÍMITE
# ============================================================================

class TestEdgeCases:
    """Tests para casos límite."""
    
    def test_empty_string(self, detector):
        """Verifica manejo de string vacío."""
        result = detector.predict("")
        assert result is not None
        assert "label" in result
    
    def test_very_long_text(self, detector):
        """Verifica manejo de texto muy largo."""
        long_text = "word " * 500
        result = detector.predict(long_text)
        assert result is not None
    
    def test_special_characters(self, detector):
        """Verifica manejo de caracteres especiales."""
        text = "Hello!  @user #hashtag https://url.com 😀"
        result = detector.predict(text)
        assert result is not None
    
    def test_unicode_text(self, detector):
        """Verifica manejo de texto unicode."""
        text = "Hola mundo 你好 مرحبا"
        result = detector.predict(text)
        assert result is not None
    
    def test_only_spaces(self, detector):
        """Verifica manejo de solo espacios."""
        result = detector.predict("     ")
        assert result is not None
    
    def test_all_caps(self, detector):
        """Verifica manejo de texto en mayúsculas."""
        result = detector.predict("THIS IS ALL CAPS")
        assert result is not None


# ============================================================================
# TESTS PARA THRESHOLD
# ============================================================================

class TestThreshold:
    """Tests para el umbral de decisión."""
    
    def test_threshold_in_valid_range(self, detector):
        """Verifica que el threshold está en rango válido."""
        threshold = detector.get_threshold()
        assert 0 < threshold < 1
    
    def test_threshold_affects_predictions(self, detector):
        """Verifica que el threshold afecta las predicciones."""
        threshold = detector.get_threshold()
        result = detector.predict("Test text")
        
        prob = result["probability"]
        expected_is_hate = prob >= threshold
        
        assert result["is_hate"] == expected_is_hate


# ============================================================================
# TESTS PARA GUARDADO Y CARGA
# ============================================================================

class TestSaveLoad:
    """Tests para guardado y carga del modelo."""
    
    def test_pkl_file_loadable(self, model_path):
        """Verifica que el archivo pkl se puede cargar."""
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        
        assert data is not None
        assert isinstance(data, dict)
    
    def test_loaded_model_functional(self, detector):
        """Verifica que el modelo cargado funciona."""
        result = detector. predict("Test prediction")
        assert result is not None
        assert "label" in result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])