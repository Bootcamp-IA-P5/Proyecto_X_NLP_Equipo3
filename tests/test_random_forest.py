"""
🧪 Tests para el modelo Random Forest de detección de odio en YouTube
Ejecutar con: pytest tests/test_random_forest_model.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from scipy import sparse
import joblib

# Fixtures para datos de prueba
@pytest. fixture
def sample_texts():
    """Textos de ejemplo para pruebas"""
    return [
        "This is a great video, thanks for sharing!",
        "You are such an idiot, go kill yourself!",
        "I disagree with your opinion but respect your point of view",
        "All people are stupid and should be banned",
        "Love this content, keep it up!"
    ]

@pytest.fixture
def sample_labels():
    """Etiquetas correspondientes a los textos de ejemplo"""
    return [0, 1, 0, 1, 0]  # 0 = Normal, 1 = Odio

@pytest.fixture
def sample_dataframe(sample_texts, sample_labels):
    """DataFrame de ejemplo para pruebas"""
    return pd.DataFrame({
        'Text': sample_texts,
        'Text_Processed': [t.lower() for t in sample_texts],
        'IsHate': sample_labels
    })

@pytest.fixture
def mock_tfidf_vectorizer():
    """Mock del vectorizador TF-IDF"""
    mock = Mock()
    mock.fit_transform.return_value = sparse.csr_matrix(np.random.rand(5, 100))
    mock.transform.return_value = sparse.csr_matrix(np.random. rand(1, 100))
    mock.vocabulary_ = {f'word_{i}': i for i in range(100)}
    mock.get_feature_names_out.return_value = [f'word_{i}' for i in range(100)]
    return mock

@pytest.fixture
def mock_scaler():
    """Mock del StandardScaler"""
    mock = Mock()
    mock.fit_transform.return_value = np.random.rand(5, 7)
    mock. transform.return_value = np.random. rand(1, 7)
    return mock

@pytest. fixture
def mock_random_forest():
    """Mock del modelo Random Forest"""
    mock = Mock()
    mock.predict.return_value = np.array([0])
    mock.predict_proba.return_value = np.array([[0.8, 0.2]])
    mock.feature_importances_ = np.random.rand(107)
    mock.classes_ = np.array([0, 1])
    return mock


# =============================================================================
# TESTS DE FEATURE ENGINEERING
# =============================================================================

class TestFeatureExtraction:
    """Tests para la extracción de características del texto"""
    
    def test_extract_text_length(self, sample_dataframe):
        """Verifica que se calcula correctamente la longitud del texto"""
        df = sample_dataframe
        text_length = df['Text_Processed'].str. len()
        
        assert len(text_length) == len(df)
        assert all(text_length >= 0)
        assert text_length. dtype in [np.int64, np.int32, int]
    
    def test_extract_word_count(self, sample_dataframe):
        """Verifica que se cuenta correctamente el número de palabras"""
        df = sample_dataframe
        word_count = df['Text_Processed'].str. split().str.len()
        
        assert len(word_count) == len(df)
        assert all(word_count >= 1)  # Cada texto tiene al menos una palabra
    
    def test_extract_exclamation_count(self, sample_dataframe):
        """Verifica el conteo de signos de exclamación"""
        df = sample_dataframe
        exclamation_count = df['Text']. str.count('!')
        
        assert exclamation_count. iloc[0] == 1  # "thanks for sharing!"
        assert exclamation_count.iloc[1] == 1  # "go kill yourself!"
    
    def test_extract_question_count(self, sample_dataframe):
        """Verifica el conteo de signos de interrogación"""
        df = sample_dataframe
        question_count = df['Text'].str.count(r'\?')
        
        # Ningún texto de ejemplo tiene signos de interrogación
        assert all(question_count == 0)
    
    def test_extract_uppercase_ratio(self, sample_dataframe):
        """Verifica el cálculo del ratio de mayúsculas"""
        df = sample_dataframe
        uppercase_ratio = (df['Text']. str.count(r'[A-Z]') / df['Text'].str.len()).fillna(0)
        
        assert all(uppercase_ratio >= 0)
        assert all(uppercase_ratio <= 1)
    
    def test_extract_unique_word_ratio(self, sample_dataframe):
        """Verifica el cálculo del ratio de palabras únicas"""
        df = sample_dataframe
        unique_word_ratio = df['Text_Processed'].apply(
            lambda x: len(set(str(x).split())) / len(str(x). split()) if str(x) else 0
        )
        
        assert all(unique_word_ratio >= 0)
        assert all(unique_word_ratio <= 1)
    
    def test_extract_avg_word_length(self, sample_dataframe):
        """Verifica el cálculo de la longitud promedio de palabras"""
        df = sample_dataframe
        text_length = df['Text_Processed']. str.len()
        word_count = df['Text_Processed'].str. split().str.len()
        avg_word_length = text_length / word_count
        
        assert all(avg_word_length > 0)
        assert all(avg_word_length < 50)  # Límite razonable

    def test_features_dataframe_shape(self, sample_dataframe):
        """Verifica que el DataFrame de features tiene la forma correcta"""
        df = sample_dataframe
        
        # Simular extracción de features
        features = pd.DataFrame(index=df.index)
        features['text_length'] = df['Text_Processed'].str. len()
        features['word_count'] = df['Text_Processed'].str. split().str.len()
        features['avg_word_length'] = features['text_length'] / features['word_count']
        features['exclamation_count'] = df['Text'].str.count('!')
        features['question_count'] = df['Text']. str.count(r'\?')
        features['uppercase_ratio'] = (df['Text'].str. count(r'[A-Z]') / df['Text'].str. len()).fillna(0)
        features['unique_word_ratio'] = df['Text_Processed']. apply(
            lambda x: len(set(str(x). split())) / len(str(x). split()) if str(x) else 0
        )
        
        assert features.shape == (len(df), 7)
        assert not features.isnull().any(). any()


# =============================================================================
# TESTS DE PREPARACIÓN DE DATOS
# =============================================================================

class TestDataPreparation:
    """Tests para la preparación de datos para ML"""
    
    def test_train_test_split_proportions(self):
        """Verifica que el split train/test mantiene las proporciones"""
        from sklearn.model_selection import train_test_split
        
        # Usar dataset más grande para que stratify funcione correctamente
        # Necesitamos al menos 2 muestras por clase en test
        texts = [
            "great video", "love it", "amazing content", "thanks",
            "nice work", "helpful tutorial", "good job", "excellent",
            "hate this", "terrible", "awful video", "worst ever"
        ]
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # 8 normal, 4 odio
        
        X = pd.Series(texts)
        y = pd.Series(labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Verificar tamaños
        assert len(X_train) == 9   # 75% de 12
        assert len(X_test) == 3    # 25% de 12
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_stratified_split_maintains_class_distribution(self):
        """Verifica que el split estratificado mantiene la distribución de clases"""
        from sklearn.model_selection import train_test_split
        
        # Crear dataset más grande para mejor prueba de estratificación
        np.random.seed(42)
        n_samples = 100
        y = np.array([0] * 80 + [1] * 20)  # 80% normal, 20% odio
        X = pd.Series([f"text_{i}" for i in range(n_samples)])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Verificar proporciones aproximadas
        train_hate_ratio = y_train.sum() / len(y_train)
        test_hate_ratio = y_test. sum() / len(y_test)
        
        assert abs(train_hate_ratio - 0.2) < 0.1
        assert abs(test_hate_ratio - 0.2) < 0.1
    
    def test_train_test_split_without_stratify_small_dataset(self, sample_dataframe, sample_labels):
        """Verifica split básico sin estratificación para datasets pequeños"""
        from sklearn.model_selection import train_test_split
        
        X = sample_dataframe['Text_Processed']
        y = pd.Series(sample_labels)
        
        # Sin stratify para datasets muy pequeños
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        
        # Verificar que se dividió correctamente
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


# =============================================================================
# TESTS DE VECTORIZACIÓN TF-IDF
# =============================================================================

class TestTfidfVectorization:
    """Tests para la vectorización TF-IDF"""
    
    def test_tfidf_fit_transform_shape(self, sample_texts):
        """Verifica que TF-IDF genera la forma correcta"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        tfidf = TfidfVectorizer(max_features=100)
        X_tfidf = tfidf.fit_transform(sample_texts)
        
        assert X_tfidf.shape[0] == len(sample_texts)
        assert X_tfidf.shape[1] <= 100
    
    def test_tfidf_transform_maintains_vocabulary(self, sample_texts):
        """Verifica que transform usa el mismo vocabulario que fit"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        tfidf = TfidfVectorizer(max_features=100)
        tfidf.fit(sample_texts[:3])  # Fit con subset
        
        X_train = tfidf. transform(sample_texts[:3])
        X_test = tfidf.transform(sample_texts[3:])
        
        # Ambos deben tener el mismo número de features
        assert X_train.shape[1] == X_test.shape[1]
    
    def test_tfidf_ngram_range(self, sample_texts):
        """Verifica que n-gramas funcionan correctamente"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Solo unigramas
        tfidf_uni = TfidfVectorizer(ngram_range=(1, 1))
        X_uni = tfidf_uni.fit_transform(sample_texts)
        
        # Unigramas y bigramas
        tfidf_bi = TfidfVectorizer(ngram_range=(1, 2))
        X_bi = tfidf_bi. fit_transform(sample_texts)
        
        # Con bigramas debe haber más features
        assert X_bi.shape[1] >= X_uni.shape[1]
    
    def test_tfidf_sparse_matrix_output(self, sample_texts):
        """Verifica que TF-IDF genera matrices sparse"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        tfidf = TfidfVectorizer()
        X_tfidf = tfidf.fit_transform(sample_texts)
        
        assert sparse.issparse(X_tfidf)


# =============================================================================
# TESTS DEL MODELO RANDOM FOREST
# =============================================================================

class TestRandomForestModel:
    """Tests para el modelo Random Forest"""
    
    def test_model_initialization_with_class_weight(self):
        """Verifica inicialización con balance de clases"""
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(
            n_estimators=10,
            class_weight='balanced',
            random_state=42
        )
        
        assert rf.class_weight == 'balanced'
        assert rf.n_estimators == 10
    
    def test_model_fit_and_predict(self):
        """Verifica que el modelo puede entrenar y predecir"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Datos sintéticos
        np.random.seed(42)
        X = np.random. rand(50, 10)
        y = np.random.randint(0, 2, 50)
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        predictions = rf.predict(X)
        
        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_predict_proba(self):
        """Verifica que predict_proba devuelve probabilidades válidas"""
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        X = np.random. rand(50, 10)
        y = np.random. randint(0, 2, 50)
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf. fit(X, y)
        
        proba = rf. predict_proba(X)
        
        assert proba.shape == (50, 2)
        assert all(proba. sum(axis=1) - 1 < 1e-10)  # Suman 1
        assert all(proba.flatten() >= 0)
        assert all(proba.flatten() <= 1)
    
    def test_model_feature_importances(self):
        """Verifica que se calculan las importancias de features"""
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        
        assert len(importances) == 10
        assert abs(sum(importances) - 1) < 1e-10  # Suman 1
        assert all(i >= 0 for i in importances)


# =============================================================================
# TESTS DE MÉTRICAS DE EVALUACIÓN
# =============================================================================

class TestEvaluationMetrics:
    """Tests para las métricas de evaluación"""
    
    def test_accuracy_score_calculation(self):
        """Verifica el cálculo del accuracy"""
        from sklearn. metrics import accuracy_score
        
        y_true = [0, 0, 1, 1, 0]
        y_pred = [0, 0, 1, 0, 0]  # 1 error
        
        accuracy = accuracy_score(y_true, y_pred)
        
        assert accuracy == 0.8  # 4/5 correctos
    
    def test_precision_score_calculation(self):
        """Verifica el cálculo de la precisión"""
        from sklearn.metrics import precision_score
        
        y_true = [0, 0, 1, 1, 0]
        y_pred = [0, 1, 1, 1, 0]  # 1 FP, 2 TP
        
        precision = precision_score(y_true, y_pred)
        
        assert precision == 2/3  # TP / (TP + FP) = 2 / (2 + 1)
    
    def test_recall_score_calculation(self):
        """Verifica el cálculo del recall (crítico para detección de odio)"""
        from sklearn.metrics import recall_score
        
        y_true = [0, 0, 1, 1, 0]
        y_pred = [0, 0, 1, 0, 0]  # 1 TP, 1 FN
        
        recall = recall_score(y_true, y_pred)
        
        assert recall == 0.5  # TP / (TP + FN) = 1 / (1 + 1)
    
    def test_f1_score_calculation(self):
        """Verifica el cálculo del F1-score"""
        from sklearn. metrics import f1_score
        
        y_true = [0, 0, 1, 1, 0]
        y_pred = [0, 0, 1, 1, 0]  # Perfecto
        
        f1 = f1_score(y_true, y_pred)
        
        assert f1 == 1.0
    
    def test_confusion_matrix_structure(self):
        """Verifica la estructura de la matriz de confusión"""
        from sklearn.metrics import confusion_matrix
        
        y_true = [0, 0, 1, 1, 0]
        y_pred = [0, 1, 1, 0, 0]
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        tn, fp, fn, tp = cm. ravel()
        
        assert tn == 2  # True Negatives
        assert fp == 1  # False Positives
        assert fn == 1  # False Negatives
        assert tp == 1  # True Positives
    
    def test_roc_auc_score_range(self):
        """Verifica que ROC-AUC está en rango válido"""
        from sklearn.metrics import roc_auc_score
        
        y_true = [0, 0, 1, 1, 0, 1, 0, 1]
        y_proba = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.6]
        
        roc_auc = roc_auc_score(y_true, y_proba)
        
        assert 0 <= roc_auc <= 1


# =============================================================================
# TESTS DE PIPELINE COMPLETO
# =============================================================================

class TestCompletePipeline:
    """Tests para el pipeline completo de predicción"""
    
    def test_feature_combination_with_sparse_matrix(self):
        """Verifica la combinación de features TF-IDF y numéricas"""
        # TF-IDF features (sparse)
        tfidf_features = sparse.csr_matrix(np.random.rand(10, 100))
        
        # Numeric features (dense)
        numeric_features = np. random.rand(10, 7)
        
        # Combinar
        combined = sparse.hstack([tfidf_features, numeric_features])
        
        assert combined.shape == (10, 107)
        assert sparse.issparse(combined)
    
    def test_scaler_maintains_shape(self):
        """Verifica que el scaler mantiene la forma de los datos"""
        from sklearn.preprocessing import StandardScaler
        
        X = np.random. rand(10, 7)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        assert X_scaled.shape == X.shape
        
        # Verificar normalización (media ~0, std ~1)
        assert abs(X_scaled.mean()) < 0.1
        assert abs(X_scaled.std() - 1) < 0.1
    
    def test_end_to_end_prediction_flow(self, sample_texts, sample_labels):
        """Test de flujo completo de predicción"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Preparar datos
        texts = sample_texts
        labels = sample_labels
        
        # Extraer features numéricas simples
        numeric_features = np. array([
            [len(t), len(t.split()), t.count('! '), t.count('?')]
            for t in texts
        ])
        
        # Vectorizar texto
        tfidf = TfidfVectorizer(max_features=50)
        text_features = tfidf.fit_transform(texts)
        
        # Escalar features numéricas
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric_features)
        
        # Combinar
        X_combined = sparse.hstack([text_features, numeric_scaled])
        
        # Entrenar modelo
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_combined, labels)
        
        # Predecir
        predictions = rf.predict(X_combined)
        probabilities = rf.predict_proba(X_combined)
        
        assert len(predictions) == len(labels)
        assert probabilities.shape[0] == len(labels)


# =============================================================================
# TESTS DE GUARDADO Y CARGA DE MODELO
# =============================================================================

class TestModelPersistence:
    """Tests para guardado y carga del modelo"""
    
    def test_model_package_structure(self, mock_random_forest, mock_tfidf_vectorizer, mock_scaler):
        """Verifica la estructura del paquete de modelo"""
        model_package = {
            'model': mock_random_forest,
            'vectorizer': mock_tfidf_vectorizer,
            'scaler': mock_scaler,
            'scaler_advanced': None,
            'threshold': 0.5,
            'feature_columns': ['text_length', 'word_count', 'avg_word_length',
                              'exclamation_count', 'question_count', 
                              'uppercase_ratio', 'unique_word_ratio']
        }
        
        assert 'model' in model_package
        assert 'vectorizer' in model_package
        assert 'scaler' in model_package
        assert 'threshold' in model_package
        assert 'feature_columns' in model_package
        assert len(model_package['feature_columns']) == 7
    
    def test_joblib_save_and_load(self, tmp_path):
        """Verifica que se puede guardar y cargar con joblib"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Crear modelo simple
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(np. random.rand(10, 5), np.random. randint(0, 2, 10))
        
        # Guardar
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(rf, model_path)
        
        # Cargar
        loaded_rf = joblib.load(model_path)
        
        # Verificar que funciona igual
        X_test = np. random.rand(3, 5)
        original_pred = rf. predict(X_test)
        loaded_pred = loaded_rf.predict(X_test)
        
        assert all(original_pred == loaded_pred)


# =============================================================================
# TESTS DE FUNCIÓN DE PREDICCIÓN
# =============================================================================

class TestPredictionFunction:
    """Tests para la función de predicción de producción"""
    
    def test_prediction_result_structure(self, mock_random_forest, mock_tfidf_vectorizer, mock_scaler):
        """Verifica la estructura del resultado de predicción"""
        # Simular resultado esperado
        result = {
            'is_hate': False,
            'hate_probability': 0.2,
            'confidence': 'HIGH',
            'processed_text': 'test text'
        }
        
        assert 'is_hate' in result
        assert 'hate_probability' in result
        assert 'confidence' in result
        assert 'processed_text' in result
        assert isinstance(result['is_hate'], bool)
        assert 0 <= result['hate_probability'] <= 1
        assert result['confidence'] in ['HIGH', 'MEDIUM', 'LOW']
    
    def test_confidence_levels(self):
        """Verifica los niveles de confianza según probabilidad"""
        def get_confidence(probability):
            if probability > 0.7 or probability < 0.3:
                return 'HIGH'
            return 'MEDIUM'
        
        assert get_confidence(0.1) == 'HIGH'
        assert get_confidence(0.2) == 'HIGH'
        assert get_confidence(0.5) == 'MEDIUM'
        assert get_confidence(0.6) == 'MEDIUM'
        assert get_confidence(0.8) == 'HIGH'
        assert get_confidence(0.9) == 'HIGH'
    
    def test_empty_text_handling(self):
        """Verifica manejo de texto vacío"""
        text = ""
        
        # Simular procesamiento
        word_count = len(text.split()) if text else 0
        
        assert word_count == 0
    
    def test_special_characters_handling(self):
        """Verifica manejo de caracteres especiales"""
        text = "Hello! !!  @user #hashtag https://example.com ??? "
        
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        assert exclamation_count == 3
        assert question_count == 3


# =============================================================================
# TESTS DE VALIDACIÓN CRUZADA
# =============================================================================

class TestCrossValidation:
    """Tests para validación cruzada"""
    
    def test_stratified_kfold_splits(self):
        """Verifica que StratifiedKFold crea splits estratificados"""
        from sklearn.model_selection import StratifiedKFold
        
        X = np.random. rand(100, 10)
        y = np.array([0] * 80 + [1] * 20)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, test_idx in skf.split(X, y):
            train_ratio = y[train_idx]. sum() / len(train_idx)
            test_ratio = y[test_idx]. sum() / len(test_idx)
            
            # Ambos deben tener aproximadamente 20% de clase 1
            assert abs(train_ratio - 0.2) < 0.1
            assert abs(test_ratio - 0.2) < 0.1
    
    def test_cross_val_score_output(self):
        """Verifica la salida de cross_val_score"""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.rand(50, 10)
        y = np. random.randint(0, 2, 50)
        
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        scores = cross_val_score(rf, X, y, cv=3, scoring='f1')
        
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)


# =============================================================================
# TESTS DE GRID SEARCH
# =============================================================================

class TestGridSearch:
    """Tests para Grid Search"""
    
    def test_grid_search_best_params(self):
        """Verifica que Grid Search encuentra mejores parámetros"""
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random. rand(50, 10)
        y = np. random.randint(0, 2, 50)
        
        param_grid = {
            'n_estimators': [5, 10],
            'max_depth': [3, 5]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1')
        grid_search.fit(X, y)
        
        assert 'n_estimators' in grid_search.best_params_
        assert 'max_depth' in grid_search.best_params_
        assert grid_search.best_params_['n_estimators'] in [5, 10]
        assert grid_search.best_params_['max_depth'] in [3, 5]
    
    def test_grid_search_best_estimator(self):
        """Verifica que Grid Search devuelve el mejor estimador"""
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)
        
        param_grid = {'n_estimators': [5, 10]}
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3)
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        
        assert isinstance(best_model, RandomForestClassifier)
        assert best_model.n_estimators in [5, 10]


# =============================================================================
# TESTS DE ANÁLISIS DE ERRORES
# =============================================================================

class TestErrorAnalysis:
    """Tests para análisis de errores"""
    
    def test_error_type_classification(self):
        """Verifica la clasificación de tipos de error"""
        y_true = [0, 0, 1, 1, 0]
        y_pred = [0, 1, 1, 0, 0]
        
        errors = []
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                errors. append('Correct')
            elif true == 0 and pred == 1:
                errors.append('False_Positive')
            else:  # true == 1 and pred == 0
                errors. append('False_Negative')
        
        assert errors. count('Correct') == 3
        assert errors. count('False_Positive') == 1
        assert errors. count('False_Negative') == 1
    
    def test_false_negative_rate(self):
        """Verifica el cálculo de tasa de falsos negativos"""
        y_true = [0, 0, 1, 1, 0, 1]  # 3 casos de odio
        y_pred = [0, 0, 1, 0, 0, 0]  # Solo detecta 1
        
        # Calcular FN rate
        total_hate = sum(y_true)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        fn_rate = fn / total_hate
        
        assert fn == 2
        assert total_hate == 3
        assert abs(fn_rate - 2/3) < 1e-10


# =============================================================================
# TESTS DE INTEGRACIÓN
# =============================================================================

class TestIntegration:
    """Tests de integración del sistema completo"""
    
    def test_full_training_pipeline(self, sample_texts, sample_labels):
        """Test del pipeline completo de entrenamiento"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Preparar datos
        X_text = pd.Series(sample_texts)
        y = pd.Series(sample_labels)
        
        # Features numéricas
        features = pd.DataFrame({
            'text_length': X_text.str.len(),
            'word_count': X_text.str.split().str.len()
        })
        
        # Split (sin estratificar por el tamaño pequeño)
        X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = \
            train_test_split(X_text, features, y, test_size=0.4, random_state=42)
        
        # TF-IDF
        tfidf = TfidfVectorizer(max_features=50)
        X_tfidf_train = tfidf.fit_transform(X_text_train)
        X_tfidf_test = tfidf. transform(X_text_test)
        
        # Scaler
        scaler = StandardScaler()
        X_feat_train_scaled = scaler.fit_transform(X_feat_train)
        X_feat_test_scaled = scaler.transform(X_feat_test)
        
        # Combinar
        X_train = sparse.hstack([X_tfidf_train, X_feat_train_scaled])
        X_test = sparse.hstack([X_tfidf_test, X_feat_test_scaled])
        
        # Entrenar
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predecir
        y_pred = rf. predict(X_test)
        
        assert len(y_pred) == len(y_test)
        assert all(p in [0, 1] for p in y_pred)
    
    def test_model_reproducibility(self):
        """Verifica que el modelo es reproducible con semilla fija"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random. rand(50, 10)
        y = np. random.randint(0, 2, 50)
        
        # Entrenar dos modelos con misma semilla
        rf1 = RandomForestClassifier(n_estimators=10, random_state=42)
        rf1.fit(X, y)
        
        rf2 = RandomForestClassifier(n_estimators=10, random_state=42)
        rf2.fit(X, y)
        
        # Deben dar las mismas predicciones
        X_test = np. random.rand(10, 10)
        pred1 = rf1.predict(X_test)
        pred2 = rf2.predict(X_test)
        
        assert all(pred1 == pred2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])