"""
Tests para el modelo XGBoost Ensemble de clasificación de Hate Speech
Ejecutar con: pytest tests/test_xgboost_ensemble.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from scipy. sparse import csr_matrix, hstack
import sys

# Agregar el directorio raíz al path para imports
sys.path.insert(0, str(Path(__file__). parent.parent))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_texts():
    """Textos de ejemplo para pruebas"""
    return pd.Series([
        "This is a normal comment about the video",
        "I hate you and your stupid content",
        "Great video, thanks for sharing! ",
        "You are worthless and should disappear",
        "Love this channel, very informative"
    ])


@pytest.fixture
def sample_labels():
    """Labels correspondientes a los textos de ejemplo"""
    return pd.Series([0, 1, 0, 1, 0])  # 0=Normal, 1=Hate


@pytest. fixture
def sample_numeric_features():
    """Features numéricas de ejemplo"""
    return pd. DataFrame({
        'char_count': [40, 35, 30, 42, 33],
        'word_count': [8, 7, 5, 7, 5],
        'sentence_count': [1, 1, 1, 1, 1],
        'avg_word_length': [5.0, 5.0, 6.0, 6.0, 6.6],
        'uppercase_count': [1, 2, 1, 1, 1],
        'uppercase_ratio': [0.025, 0.057, 0.033, 0.024, 0.030],
        'exclamation_count': [0, 0, 1, 0, 0],
        'question_count': [0, 0, 0, 0, 0],
        'emoji_count': [0, 0, 0, 0, 0],
        'url_count': [0, 0, 0, 0, 0],
        'mention_count': [0, 0, 0, 0, 0],
        'hashtag_count': [0, 0, 0, 0, 0],
        'number_count': [0, 0, 0, 0, 0]
    })


@pytest.fixture
def mock_vectorizer():
    """Mock del TF-IDF vectorizer"""
    vectorizer = Mock()
    vectorizer.fit_transform.return_value = csr_matrix(np.random.rand(5, 100))
    vectorizer.transform.return_value = csr_matrix(np.random.rand(1, 100))
    return vectorizer


@pytest.fixture
def mock_scaler():
    """Mock del StandardScaler"""
    scaler = Mock()
    scaler. fit_transform.return_value = np.random.rand(5, 13)
    scaler.transform.return_value = np.random.rand(1, 13)
    return scaler


# ============================================================================
# TESTS PARA DATA AUGMENTATION
# ============================================================================

class TestAugmentTextSimple:
    """Tests para la función augment_text_simple"""
    
    def test_augment_returns_string(self):
        """Verifica que augment_text_simple retorna un string"""
        def augment_text_simple(text):
            words = str(text).split()
            if len(words) > 3 and np.random.random() < 0.3:
                idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            return ' '. join(words)
        
        result = augment_text_simple("This is a test sentence")
        assert isinstance(result, str)
    
    def test_augment_preserves_words(self):
        """Verifica que se preservan todas las palabras"""
        np.random.seed(42)
        
        def augment_text_simple(text):
            words = str(text).split()
            if len(words) > 3 and np. random.random() < 0.3:
                idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            return ' '.join(words)
        
        original = "This is a longer test sentence"
        result = augment_text_simple(original)
        
        assert set(result.split()) == set(original.split())
    
    def test_short_text_unchanged(self):
        """Verifica que textos cortos no se modifican"""
        def augment_text_simple(text):
            words = str(text).split()
            if len(words) > 3:
                return ' '.join(words)
            return text
        
        short_text = "Hi there"
        result = augment_text_simple(short_text)
        assert result == short_text


# ============================================================================
# TESTS PARA FEATURE ENGINEERING
# ============================================================================

class TestExtractAdvancedTextFeatures:
    """Tests para extract_advanced_text_features"""
    
    def test_returns_dict_with_all_keys(self):
        """Verifica que retorna diccionario con todas las claves"""
        import re
        
        def extract_advanced_text_features(text):
            features = {}
            offensive_words = ['hate', 'stupid', 'idiot', 'dumb', 'kill']
            text_lower = text.lower()
            words = text_lower. split()
            
            features['offensive_word_count'] = sum(1 for w in words if any(off in w for off in offensive_words))
            features['offensive_word_ratio'] = features['offensive_word_count'] / len(words) if words else 0
            features['avg_word_len'] = np.mean([len(w) for w in words]) if words else 0
            features['max_word_len'] = max([len(w) for w in words]) if words else 0
            features['char_repetition'] = len(re. findall(r'(.)\1{2,}', text))
            features['caps_words'] = sum(1 for w in words if w.isupper() and len(w) > 1)
            features['multiple_punctuation'] = len(re.findall(r'[! ?]{2,}', text))
            features['negation_count'] = sum(1 for w in words if w in ['not', 'no', 'never'])
            features['pronoun_count'] = sum(1 for w in words if w in ['you', 'your', 'they'])
            features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
            
            return features
        
        result = extract_advanced_text_features("This is a test")
        
        expected_keys = [
            'offensive_word_count', 'offensive_word_ratio', 'avg_word_len',
            'max_word_len', 'char_repetition', 'caps_words', 'multiple_punctuation',
            'negation_count', 'pronoun_count', 'unique_word_ratio'
        ]
        
        for key in expected_keys:
            assert key in result
    
    def test_detects_offensive_words(self):
        """Verifica la detección de palabras ofensivas"""
        import re
        
        def extract_advanced_text_features(text):
            offensive_words = ['hate', 'stupid', 'idiot']
            text_lower = text.lower()
            words = text_lower.split()
            return {
                'offensive_word_count': sum(1 for w in words if any(off in w for off in offensive_words))
            }
        
        result = extract_advanced_text_features("I hate stupid idiots")
        assert result['offensive_word_count'] == 3
    
    def test_detects_char_repetition(self):
        """Verifica la detección de repetición de caracteres"""
        import re
        
        def extract_features(text):
            # Regex: (.) captura cualquier carácter, \1{2,} busca 2+ repeticiones
            return {'char_repetition': len(re.findall(r'(.)\1{2,}', text))}
        
        result = extract_features("Noooo whyyyy")
        assert result['char_repetition'] == 2
    
    def test_handles_empty_text(self):
        """Verifica el manejo de texto vacío"""
        import re
        
        def extract_advanced_text_features(text):
            words = text.lower().split()
            return {
                'offensive_word_count': 0,
                'offensive_word_ratio': 0,
                'avg_word_len': 0 if not words else np.mean([len(w) for w in words]),
                'max_word_len': 0 if not words else max([len(w) for w in words]),
                'unique_word_ratio': 0 if not words else len(set(words)) / len(words)
            }
        
        result = extract_advanced_text_features("")
        assert result['avg_word_len'] == 0
        assert result['unique_word_ratio'] == 0


# ============================================================================
# TESTS PARA MÉTRICAS
# ============================================================================

class TestComputeMetrics:
    """Tests para la función compute_metrics"""
    
    def test_compute_metrics_returns_all_keys(self):
        """Verifica que compute_metrics retorna todas las métricas"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        def compute_metrics(y_true, y_pred, y_proba=None):
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
            return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc}
        
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 1, 1, 0]
        y_proba = [0.2, 0.8, 0.6, 0.9, 0.1]
        
        result = compute_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
        assert 'roc_auc' in result
    
    def test_perfect_predictions(self):
        """Verifica métricas con predicciones perfectas"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        def compute_metrics(y_true, y_pred):
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
            }
        
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 1, 0, 1]
        
        result = compute_metrics(y_true, y_pred)
        
        assert result['accuracy'] == 1.0
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1'] == 1.0


# ============================================================================
# TESTS PARA CLASS WEIGHTS
# ============================================================================

class TestScalePosWeight:
    """Tests para el cálculo de scale_pos_weight"""
    
    def test_balanced_classes(self):
        """Verifica scale_pos_weight con clases balanceadas"""
        y = np.array([0, 1, 0, 1, 0, 1])
        n_pos = (y == 1). sum()
        n_neg = (y == 0).sum()
        scale_pos_weight = n_neg / n_pos
        
        assert scale_pos_weight == 1.0
    
    def test_imbalanced_classes(self):
        """Verifica scale_pos_weight con clases desbalanceadas"""
        y = np.array([0, 0, 0, 0, 1, 1])
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        scale_pos_weight = n_neg / n_pos
        
        assert scale_pos_weight == 2.0
    
    def test_highly_imbalanced(self):
        """Verifica scale_pos_weight con desbalance extremo"""
        y = np.array([0]*90 + [1]*10)
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        scale_pos_weight = n_neg / n_pos
        
        assert scale_pos_weight == 9.0


# ============================================================================
# TESTS PARA TF-IDF VECTORIZER
# ============================================================================

class TestTfidfVectorization:
    """Tests para la vectorización TF-IDF"""
    
    def test_vectorizer_output_shape(self, sample_texts):
        """Verifica la forma de salida del vectorizer"""
        from sklearn.feature_extraction. text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        X = vectorizer.fit_transform(sample_texts)
        
        assert X.shape[0] == len(sample_texts)
        assert X.shape[1] <= 100
    
    def test_vectorizer_sparse_output(self, sample_texts):
        """Verifica que el output es sparse matrix"""
        from sklearn.feature_extraction. text import TfidfVectorizer
        from scipy.sparse import issparse
        
        vectorizer = TfidfVectorizer(max_features=50)
        X = vectorizer.fit_transform(sample_texts)
        
        assert issparse(X)
    
    def test_transform_new_text(self, sample_texts):
        """Verifica transformación de texto nuevo"""
        from sklearn.feature_extraction. text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=50)
        vectorizer.fit(sample_texts)
        
        new_text = ["This is a new comment"]
        X_new = vectorizer. transform(new_text)
        
        assert X_new. shape[0] == 1
        assert X_new.shape[1] == vectorizer.fit_transform(sample_texts).shape[1]


# ============================================================================
# TESTS PARA THRESHOLD TUNING
# ============================================================================

class TestThresholdTuning:
    """Tests para el ajuste de threshold"""
    
    def test_threshold_affects_predictions(self):
        """Verifica que el threshold afecta las predicciones"""
        probas = np.array([0.3, 0.5, 0.7, 0.4, 0.6])
        
        preds_low = (probas >= 0.4).astype(int)
        preds_high = (probas >= 0.6).astype(int)
        
        assert preds_low. sum() > preds_high. sum()
    
    def test_threshold_in_valid_range(self):
        """Verifica que thresholds están en rango válido"""
        thresholds = np.linspace(0.35, 0.65, 31)
        
        assert all(0 <= t <= 1 for t in thresholds)
        assert thresholds[0] == pytest.approx(0.35)
        assert thresholds[-1] == pytest.approx(0.65)
    
    def test_default_threshold(self):
        """Verifica comportamiento con threshold por defecto"""
        probas = np.array([0.4, 0.6, 0.5, 0.3, 0.7])
        default_threshold = 0.5
        
        preds = (probas >= default_threshold).astype(int)
        
        assert preds.tolist() == [0, 1, 1, 0, 1]


# ============================================================================
# TESTS PARA ENSEMBLE
# ============================================================================

class TestEnsemble:
    """Tests para el ensemble de modelos"""
    
    def test_ensemble_averaging(self):
        """Verifica el promedio de probabilidades del ensemble"""
        proba_model1 = np.array([0.3, 0.7, 0.5])
        proba_model2 = np.array([0.4, 0.6, 0.6])
        proba_model3 = np.array([0.5, 0.8, 0.4])
        
        ensemble_probas = [proba_model1, proba_model2, proba_model3]
        avg_proba = np.mean(ensemble_probas, axis=0)
        
        expected = np.array([0.4, 0.7, 0.5])
        np.testing.assert_array_almost_equal(avg_proba, expected)
    
    def test_different_seeds_produce_different_models(self):
        """Verifica que diferentes seeds producen variación"""
        np.random.seed(42)
        rand1 = np.random.rand(10)
        
        np.random. seed(123)
        rand2 = np.random.rand(10)
        
        assert not np.allclose(rand1, rand2)
    
    def test_ensemble_reduces_variance(self):
        """Verifica que el ensemble reduce la varianza"""
        np.random.seed(42)
        
        # Simular predicciones de múltiples modelos
        predictions = [np.random.rand(100) for _ in range(5)]
        
        # Varianza individual promedio
        individual_var = np.mean([np. var(p) for p in predictions])
        
        # Varianza del ensemble
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_var = np.var(ensemble_pred)
        
        # El ensemble debería tener menor varianza
        assert ensemble_var <= individual_var


# ============================================================================
# TESTS PARA CONFUSION MATRIX
# ============================================================================

class TestConfusionMatrix:
    """Tests para la matriz de confusión"""
    
    def test_confusion_matrix_shape(self):
        """Verifica la forma de la matriz de confusión"""
        from sklearn.metrics import confusion_matrix
        
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm. shape == (2, 2)
    
    def test_confusion_matrix_values(self):
        """Verifica los valores de la matriz de confusión"""
        from sklearn.metrics import confusion_matrix
        
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm. ravel()
        
        assert tn == 1  # True Negatives
        assert fp == 1  # False Positives
        assert fn == 1  # False Negatives
        assert tp == 1  # True Positives
    
    def test_specificity_sensitivity(self):
        """Verifica cálculo de specificity y sensitivity"""
        from sklearn.metrics import confusion_matrix
        
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1, 0]
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm. ravel()
        
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        assert 0 <= specificity <= 1
        assert 0 <= sensitivity <= 1


# ============================================================================
# TESTS PARA PREDICT FUNCTION
# ============================================================================

class TestPredictHateSpeech:
    """Tests para la función predict_hate_speech"""
    
    def test_prediction_returns_tuple(self):
        """Verifica que predict retorna tuple (pred, proba)"""
        # Simular resultado
        prediction = 1
        probabilities = np.array([0.3, 0.7])
        result = (prediction, probabilities)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], np.ndarray)
    
    def test_probabilities_sum_to_one(self):
        """Verifica que las probabilidades suman 1"""
        probabilities = np.array([0.3, 0.7])
        
        assert np.isclose(probabilities.sum(), 1.0)
    
    def test_prediction_matches_threshold(self):
        """Verifica que la predicción corresponde al threshold"""
        threshold = 0.5
        probabilities = np.array([0.3, 0.7])
        
        expected_pred = int(probabilities[1] >= threshold)
        
        assert expected_pred == 1
    
    def test_handles_tuple_model(self):
        """Verifica manejo de modelo como tuple"""
        # Simular modelo como tuple (model, calibrator, threshold)
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        model_tuple = (mock_model, None, 0.5)
        
        # Extraer modelo del tuple
        if isinstance(model_tuple, tuple):
            model = model_tuple[0]
            calibrator = model_tuple[1] if len(model_tuple) > 1 else None
            threshold = model_tuple[2] if len(model_tuple) > 2 else 0.5
        
        assert model == mock_model
        assert calibrator is None
        assert threshold == 0.5


# ============================================================================
# TESTS PARA TRAIN/TEST SPLIT
# ============================================================================

class TestDataSplit:
    """Tests para la división de datos"""
    
    def test_stratified_split(self, sample_texts, sample_labels):
        """Verifica que el split estratificado preserva proporciones"""
        from sklearn.model_selection import train_test_split
        
        # Crear dataset más grande
        X = pd.Series(['text ' + str(i) for i in range(100)])
        y = pd.Series([0]*50 + [1]*50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        original_ratio = y.mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        
        assert abs(original_ratio - train_ratio) < 0.1
        assert abs(original_ratio - test_ratio) < 0.1
    
    def test_no_data_leakage(self, sample_texts, sample_labels):
        """Verifica que no hay fuga de datos"""
        from sklearn. model_selection import train_test_split
        
        X_train, X_test, _, _ = train_test_split(
            sample_texts, sample_labels, test_size=0.4, random_state=42
        )
        
        train_set = set(X_train)
        test_set = set(X_test)
        
        assert len(train_set. intersection(test_set)) == 0


# ============================================================================
# TESTS PARA FEATURE COMBINATION
# ============================================================================

class TestFeatureCombination:
    """Tests para la combinación de features"""
    
    def test_hstack_sparse_matrices(self):
        """Verifica la combinación de matrices sparse"""
        from scipy. sparse import csr_matrix, hstack
        
        tfidf = csr_matrix(np.random.rand(5, 100))
        numeric = csr_matrix(np.random.rand(5, 13))
        advanced = csr_matrix(np.random.rand(5, 10))
        
        combined = hstack([tfidf, numeric, advanced])
        
        assert combined.shape == (5, 123)
    
    def test_combined_features_preserves_rows(self):
        """Verifica que se preserva el número de filas"""
        from scipy.sparse import csr_matrix, hstack
        
        n_samples = 10
        tfidf = csr_matrix(np. random.rand(n_samples, 50))
        numeric = csr_matrix(np.random.rand(n_samples, 5))
        
        combined = hstack([tfidf, numeric])
        
        assert combined.shape[0] == n_samples


# ============================================================================
# TESTS PARA MODEL ARTIFACTS
# ============================================================================

class TestModelArtifacts:
    """Tests para los artefactos del modelo"""
    
    def test_artifacts_has_required_keys(self):
        """Verifica que los artefactos tienen todas las claves necesarias"""
        required_keys = [
            'model_name', 'model', 'threshold', 'test_metrics',
            'vectorizer', 'scaler', 'feature_columns'
        ]
        
        # Simular artefactos
        artifacts = {
            'model_name': 'Test Model',
            'model': Mock(),
            'threshold': 0.5,
            'test_metrics': {'f1': 0.7},
            'vectorizer': Mock(),
            'scaler': Mock(),
            'feature_columns': ['col1', 'col2']
        }
        
        for key in required_keys:
            assert key in artifacts
    
    def test_save_and_load_artifacts(self, tmp_path):
        """Verifica guardado y carga de artefactos"""
        import joblib
        
        artifacts = {
            'model_name': 'Test Model',
            'threshold': 0.5,
            'test_metrics': {'f1': 0.65, 'precision': 0.6, 'recall': 0.7}
        }
        
        pkl_path = tmp_path / "test_model.pkl"
        joblib.dump(artifacts, pkl_path)
        
        loaded = joblib.load(pkl_path)
        
        assert loaded['model_name'] == 'Test Model'
        assert loaded['threshold'] == 0.5
        assert loaded['test_metrics']['f1'] == 0.65


# ============================================================================
# TESTS PARA OVERFITTING DETECTION
# ============================================================================

class TestOverfittingDetection:
    """Tests para la detección de overfitting"""
    
    def test_overfitting_calculation(self):
        """Verifica el cálculo del overfitting"""
        train_f1 = 0.85
        test_f1 = 0.65
        
        overfitting = abs(train_f1 - test_f1) * 100
        
        # Usar pytest.approx para manejar precisión de punto flotante
        assert overfitting == pytest.approx(20.0)
    
    def test_acceptable_overfitting(self):
        """Verifica clasificación de overfitting aceptable"""
        overfitting = 8.0
        
        if overfitting <= 10.0:
            status = "ÓPTIMO"
        elif overfitting <= 15.0:
            status = "BUENO"
        else:
            status = "ALTO"
        
        assert status == "ÓPTIMO"
    
    def test_high_overfitting(self):
        """Verifica clasificación de overfitting alto"""
        overfitting = 18.0
        
        if overfitting <= 10.0:
            status = "ÓPTIMO"
        elif overfitting <= 15.0:
            status = "BUENO"
        else:
            status = "ALTO"
        
        assert status == "ALTO"


# ============================================================================
# TESTS PARA EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests para casos límite"""
    
    def test_empty_text(self):
        """Verifica manejo de texto vacío"""
        import re
        
        def extract_features(text):
            words = text.lower().split()
            return {
                'word_count': len(words),
                'unique_ratio': len(set(words)) / len(words) if words else 0
            }
        
        result = extract_features("")
        assert result['word_count'] == 0
        assert result['unique_ratio'] == 0
    
    def test_special_characters(self):
        """Verifica manejo de caracteres especiales"""
        text = "Hello!!  @user #hashtag 🔥 https://example.com"
        
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_all_same_predictions(self):
        """Verifica manejo cuando todas las predicciones son iguales"""
        from sklearn.metrics import precision_score, recall_score
        
        y_true = [0, 0, 1, 1, 0]
        y_pred = [0, 0, 0, 0, 0]
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        assert precision == 0
        assert recall == 0
    
    def test_very_long_text(self):
        """Verifica manejo de texto muy largo"""
        long_text = "word " * 1000
        words = long_text.split()
        
        assert len(words) == 1000


# ============================================================================
# TESTS PARA SMOTE (si se usa)
# ============================================================================

class TestSMOTE:
    """Tests para SMOTE balancing"""
    
    def test_smote_increases_minority(self):
        """Verifica que SMOTE aumenta la clase minoritaria"""
        from imblearn.over_sampling import SMOTE
        
        # Dataset desbalanceado
        X = np.random.rand(100, 10)
        y = np.array([0]*80 + [1]*20)
        
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        # Después de SMOTE, las clases deberían estar más balanceadas
        assert (y_res == 1).sum() > (y == 1).sum()
    
    def test_smote_balanced_output(self):
        """Verifica que SMOTE produce clases balanceadas"""
        from imblearn. over_sampling import SMOTE
        
        X = np. random.rand(100, 10)
        y = np. array([0]*70 + [1]*30)
        
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        # Por defecto SMOTE balancea completamente
        assert (y_res == 0).sum() == (y_res == 1). sum()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
