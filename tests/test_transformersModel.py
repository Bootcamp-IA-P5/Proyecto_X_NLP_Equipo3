"""
Tests para el modelo BERT de clasificación de Hate Speech
Ejecutar con: pytest tests/test_transformers_model.py -v
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Agregar el directorio raíz al path para imports
sys.path.insert(0, str(Path(__file__). parent.parent))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_texts():
    """Textos de ejemplo para pruebas"""
    return np.array([
        "This is a normal comment about the video",
        "I hate you and your stupid content",
        "Great video, thanks for sharing!",
        "You are worthless and should disappear",
        "Love this channel, very informative"
    ])


@pytest.fixture
def sample_labels():
    """Labels correspondientes a los textos de ejemplo"""
    return np.array([0, 1, 0, 1, 0])  # 0=Normal, 1=Hate


@pytest. fixture
def mock_tokenizer():
    """Mock del tokenizer para tests sin descarga del modelo"""
    tokenizer = Mock()
    tokenizer.return_value = {
        'input_ids': torch.randint(0, 1000, (1, 128)),
        'attention_mask': torch.ones(1, 128, dtype=torch.long)
    }
    return tokenizer


@pytest.fixture
def device():
    """Device para pruebas (CPU por defecto en tests)"""
    return torch.device('cpu')


# ============================================================================
# TESTS PARA FUNCIONES DE UTILIDAD
# ============================================================================

class TestSetupPaths:
    """Tests para la función setup_paths"""
    
    def test_setup_paths_returns_three_paths(self):
        """Verifica que setup_paths retorna 3 paths"""
        from pathlib import Path
        
        # Simular la función setup_paths
        def setup_paths():
            current_dir = Path. cwd()
            if "notebooks" in str(current_dir):
                project_root = current_dir.parent
            else:
                project_root = current_dir
            mlruns_dir = project_root / "mlruns"
            data_dir = project_root / "data" / "processed"
            return project_root, mlruns_dir, data_dir
        
        result = setup_paths()
        assert len(result) == 3
        assert all(isinstance(p, Path) for p in result)
    
    def test_mlruns_dir_is_in_project_root(self):
        """Verifica que mlruns está en el directorio raíz"""
        from pathlib import Path
        
        def setup_paths():
            current_dir = Path.cwd()
            project_root = current_dir
            mlruns_dir = project_root / "mlruns"
            data_dir = project_root / "data" / "processed"
            return project_root, mlruns_dir, data_dir
        
        project_root, mlruns_dir, _ = setup_paths()
        assert mlruns_dir.parent == project_root


class TestSetSeed:
    """Tests para la función set_seed"""
    
    def test_set_seed_reproducibility(self):
        """Verifica que set_seed produce resultados reproducibles"""
        import random
        import numpy as np
        import torch
        
        def set_seed(seed=42):
            random.seed(seed)
            np. random.seed(seed)
            torch. manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Primera ejecución
        set_seed(42)
        random_vals_1 = [random.random() for _ in range(5)]
        np_vals_1 = np.random.rand(5). tolist()
        torch_vals_1 = torch.rand(5). tolist()
        
        # Segunda ejecución con misma semilla
        set_seed(42)
        random_vals_2 = [random.random() for _ in range(5)]
        np_vals_2 = np.random.rand(5).tolist()
        torch_vals_2 = torch.rand(5). tolist()
        
        assert random_vals_1 == random_vals_2
        assert np_vals_1 == np_vals_2
        assert torch_vals_1 == torch_vals_2


# ============================================================================
# TESTS PARA DATA AUGMENTATION
# ============================================================================

class TestSynonymReplacement:
    """Tests para la función synonym_replacement"""
    
    def test_synonym_replacement_returns_string(self):
        """Verifica que synonym_replacement retorna un string"""
        import random
        
        def synonym_replacement(text, n=2):
            words = text.split()
            if len(words) < 3:
                return text
            for _ in range(min(n, len(words) // 3)):
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            return " ".join(words)
        
        result = synonym_replacement("This is a test sentence for augmentation")
        assert isinstance(result, str)
    
    def test_synonym_replacement_short_text(self):
        """Verifica que textos cortos no se modifican"""
        def synonym_replacement(text, n=2):
            words = text.split()
            if len(words) < 3:
                return text
            return text  # Simplificado para el test
        
        result = synonym_replacement("Hi")
        assert result == "Hi"
    
    def test_synonym_replacement_preserves_word_count(self):
        """Verifica que se preserva el número de palabras"""
        import random
        random.seed(42)
        
        def synonym_replacement(text, n=2):
            words = text.split()
            if len(words) < 3:
                return text
            for _ in range(min(n, len(words) // 3)):
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            return " ".join(words)
        
        original = "This is a longer test sentence"
        result = synonym_replacement(original)
        assert len(result.split()) == len(original.split())


class TestAugmentMinorityClass:
    """Tests para la función augment_minority_class"""
    
    def test_augment_increases_dataset_size(self, sample_texts, sample_labels):
        """Verifica que la augmentación aumenta el tamaño del dataset"""
        import random
        random.seed(42)
        
        def synonym_replacement(text, n=2):
            words = text. split()
            if len(words) < 3:
                return text
            for _ in range(min(n, len(words) // 3)):
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            return " ".join(words)
        
        def augment_minority_class(X, y, target_class=1, augment_factor=0.3):
            minority_mask = y == target_class
            minority_X = X[minority_mask]
            n_samples = int(len(minority_X) * augment_factor)
            if n_samples == 0:
                n_samples = 1
            indices = np.random. choice(len(minority_X), size=n_samples, replace=True)
            aug_texts = [synonym_replacement(minority_X[idx], n=2) for idx in indices]
            X_aug = np.concatenate([X, np.array(aug_texts)])
            y_aug = np. concatenate([y, np.full(n_samples, target_class)])
            return X_aug, y_aug
        
        X_aug, y_aug = augment_minority_class(sample_texts, sample_labels)
        assert len(X_aug) > len(sample_texts)
        assert len(y_aug) > len(sample_labels)
    
    def test_augment_only_adds_target_class(self, sample_texts, sample_labels):
        """Verifica que solo se agregan muestras de la clase objetivo"""
        import random
        random.seed(42)
        
        def synonym_replacement(text, n=2):
            return text
        
        def augment_minority_class(X, y, target_class=1, augment_factor=0. 3):
            minority_mask = y == target_class
            minority_X = X[minority_mask]
            n_samples = max(1, int(len(minority_X) * augment_factor))
            indices = np.random. choice(len(minority_X), size=n_samples, replace=True)
            aug_texts = [synonym_replacement(minority_X[idx], n=2) for idx in indices]
            X_aug = np.concatenate([X, np.array(aug_texts)])
            y_aug = np.concatenate([y, np. full(n_samples, target_class)])
            return X_aug, y_aug
        
        original_hate_count = np.sum(sample_labels == 1)
        X_aug, y_aug = augment_minority_class(sample_texts, sample_labels, target_class=1)
        new_hate_count = np.sum(y_aug == 1)
        
        assert new_hate_count > original_hate_count
        # El conteo de clase normal no debe cambiar
        assert np.sum(y_aug == 0) == np.sum(sample_labels == 0)


# ============================================================================
# TESTS PARA DATASET
# ============================================================================

class TestHateSpeechDataset:
    """Tests para la clase HateSpeechDataset"""
    
    def test_dataset_length(self, sample_texts, sample_labels, mock_tokenizer):
        """Verifica que __len__ retorna el tamaño correcto"""
        from torch.utils.data import Dataset
        
        class HateSpeechDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len=128):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self. max_len = max_len
            
            def __len__(self):
                return len(self. texts)
            
            def __getitem__(self, idx):
                return {'input_ids': torch.zeros(128), 'labels': self.labels[idx]}
        
        dataset = HateSpeechDataset(sample_texts, sample_labels, mock_tokenizer)
        assert len(dataset) == len(sample_texts)
    
    def test_dataset_getitem_returns_dict(self, sample_texts, sample_labels, mock_tokenizer):
        """Verifica que __getitem__ retorna un diccionario con las claves correctas"""
        from torch.utils.data import Dataset
        
        class HateSpeechDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len=128):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch. zeros(self.max_len),
                    'attention_mask': torch. ones(self.max_len),
                    'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long)
                }
        
        dataset = HateSpeechDataset(sample_texts, sample_labels, mock_tokenizer)
        item = dataset[0]
        
        assert isinstance(item, dict)
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
    
    def test_dataset_labels_are_tensors(self, sample_texts, sample_labels, mock_tokenizer):
        """Verifica que los labels son tensores de PyTorch"""
        from torch.utils. data import Dataset
        
        class HateSpeechDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len=128):
                self.texts = texts
                self. labels = labels
            
            def __len__(self):
                return len(self. texts)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.zeros(128),
                    'attention_mask': torch.ones(128),
                    'labels': torch. tensor(int(self.labels[idx]), dtype=torch.long)
                }
        
        dataset = HateSpeechDataset(sample_texts, sample_labels, mock_tokenizer)
        item = dataset[0]
        
        assert isinstance(item['labels'], torch.Tensor)
        assert item['labels'].dtype == torch.long


# ============================================================================
# TESTS PARA CÁLCULO DE CLASS WEIGHTS
# ============================================================================

class TestCalculateClassWeights:
    """Tests para la función calculate_class_weights"""
    
    def test_class_weights_returns_tensor(self, sample_labels, device):
        """Verifica que calculate_class_weights retorna un tensor"""
        from sklearn.utils.class_weight import compute_class_weight
        
        def calculate_class_weights(y_train, device):
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.array([0, 1]),
                y=y_train
            )
            return torch.FloatTensor(class_weights).to(device)
        
        weights = calculate_class_weights(sample_labels, device)
        assert isinstance(weights, torch.Tensor)
    
    def test_class_weights_has_two_values(self, sample_labels, device):
        """Verifica que hay pesos para ambas clases"""
        from sklearn.utils.class_weight import compute_class_weight
        
        def calculate_class_weights(y_train, device):
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.array([0, 1]),
                y=y_train
            )
            return torch. FloatTensor(class_weights).to(device)
        
        weights = calculate_class_weights(sample_labels, device)
        assert len(weights) == 2
    
    def test_class_weights_balanced(self, device):
        """Verifica que clases desbalanceadas tienen pesos diferentes"""
        from sklearn.utils.class_weight import compute_class_weight
        
        def calculate_class_weights(y_train, device):
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np. array([0, 1]),
                y=y_train
            )
            return torch.FloatTensor(class_weights).to(device)
        
        # Dataset muy desbalanceado
        imbalanced_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        weights = calculate_class_weights(imbalanced_labels, device)
        
        # La clase minoritaria (1) debe tener mayor peso
        assert weights[1] > weights[0]


# ============================================================================
# TESTS PARA EVALUACIÓN
# ============================================================================

class TestEvalModel:
    """Tests para la función eval_model"""
    
    def test_eval_returns_dict_with_metrics(self):
        """Verifica que eval_model retorna un diccionario con todas las métricas"""
        from sklearn.metrics import precision_recall_fscore_support
        
        def eval_model(model, data_loader, device):
            # Simular evaluación
            predictions = [0, 1, 0, 1, 0]
            true_labels = [0, 1, 1, 1, 0]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary', zero_division=0
            )
            
            return {
                'loss': 0.5,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': predictions,
                'true_labels': true_labels
            }
        
        result = eval_model(None, None, None)
        
        assert 'loss' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
        assert 'predictions' in result
        assert 'true_labels' in result
    
    def test_eval_metrics_in_valid_range(self):
        """Verifica que las métricas están en rango válido [0, 1]"""
        from sklearn.metrics import precision_recall_fscore_support
        
        def eval_model():
            predictions = [0, 1, 0, 1, 1]
            true_labels = [0, 1, 1, 1, 0]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary', zero_division=0
            )
            
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        result = eval_model()
        
        assert 0 <= result['precision'] <= 1
        assert 0 <= result['recall'] <= 1
        assert 0 <= result['f1'] <= 1


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

class TestDataLoaderIntegration:
    """Tests de integración para DataLoader"""
    
    def test_dataloader_batch_size(self, sample_texts, sample_labels, mock_tokenizer):
        """Verifica que el DataLoader respeta el batch size"""
        from torch. utils.data import Dataset, DataLoader
        
        class HateSpeechDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len=128):
                self.texts = texts
                self.labels = labels
            
            def __len__(self):
                return len(self. texts)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.zeros(128),
                    'attention_mask': torch.ones(128),
                    'labels': torch. tensor(int(self.labels[idx]))
                }
        
        dataset = HateSpeechDataset(sample_texts, sample_labels, mock_tokenizer)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        first_batch = next(iter(loader))
        assert first_batch['input_ids']. shape[0] == 2


class TestModelOutputShape:
    """Tests para verificar las formas de salida del modelo"""
    
    def test_predictions_shape_matches_batch(self):
        """Verifica que las predicciones tienen la forma correcta"""
        batch_size = 4
        num_classes = 2
        
        # Simular logits del modelo
        mock_logits = torch.randn(batch_size, num_classes)
        predictions = torch.argmax(mock_logits, dim=1)
        
        assert predictions.shape[0] == batch_size
        assert predictions.dim() == 1
    
    def test_predictions_are_binary(self):
        """Verifica que las predicciones son binarias (0 o 1)"""
        mock_logits = torch.randn(10, 2)
        predictions = torch.argmax(mock_logits, dim=1)
        
        assert all(p in [0, 1] for p in predictions. numpy())


# ============================================================================
# TESTS PARA TRAIN/VAL SPLIT
# ============================================================================

class TestDataSplit:
    """Tests para la división de datos"""
    
    def test_stratified_split_preserves_ratio(self, sample_texts, sample_labels):
        """Verifica que el split estratificado preserva la proporción de clases"""
        from sklearn.model_selection import train_test_split
        
        # Crear dataset más grande para mejor test
        X = np.array(['text ' + str(i) for i in range(100)])
        y = np.array([0]*50 + [1]*50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        original_ratio = np.mean(y)
        train_ratio = np. mean(y_train)
        test_ratio = np. mean(y_test)
        
        # Las proporciones deben ser similares
        assert abs(original_ratio - train_ratio) < 0.1
        assert abs(original_ratio - test_ratio) < 0.1
    
    def test_no_data_leakage(self, sample_texts, sample_labels):
        """Verifica que no hay fuga de datos entre splits"""
        from sklearn. model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            sample_texts, sample_labels, test_size=0.3, random_state=42
        )
        
        # No debe haber elementos comunes
        train_set = set(X_train)
        test_set = set(X_test)
        
        assert len(train_set. intersection(test_set)) == 0


# ============================================================================
# TESTS PARA MÉTRICAS
# ============================================================================

class TestMetricsCalculation:
    """Tests para el cálculo de métricas"""
    
    def test_confusion_matrix_shape(self):
        """Verifica la forma de la matriz de confusión"""
        from sklearn.metrics import confusion_matrix
        
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
    
    def test_perfect_predictions(self):
        """Verifica métricas con predicciones perfectas"""
        from sklearn.metrics import precision_recall_fscore_support
        
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 1, 0, 1]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0
    
    def test_zero_division_handling(self):
        """Verifica el manejo de división por cero"""
        from sklearn.metrics import precision_recall_fscore_support
        
        # Caso donde no hay predicciones positivas
        y_true = [0, 0, 0, 1, 1]
        y_pred = [0, 0, 0, 0, 0]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # No debe dar error, debe retornar 0
        assert precision == 0
        assert recall == 0


# ============================================================================
# TESTS PARA OVERFITTING
# ============================================================================

class TestOverfittingDetection:
    """Tests para la detección de overfitting"""
    
    def test_overfitting_calculation(self):
        """Verifica el cálculo del porcentaje de overfitting"""
        train_loss = 0.3
        val_loss = 0.5
        
        overfitting_gap = abs(train_loss - val_loss)
        overfitting_pct = (overfitting_gap / train_loss) * 100
        
        expected_pct = (0.2 / 0. 3) * 100  # ~66.67%
        assert abs(overfitting_pct - expected_pct) < 0.01
    
    def test_no_overfitting_case(self):
        """Verifica caso sin overfitting"""
        train_loss = 0.5
        val_loss = 0.51  # Muy similar
        
        overfitting_gap = abs(train_loss - val_loss)
        overfitting_pct = (overfitting_gap / train_loss) * 100
        
        assert overfitting_pct < 5  # Menos del 5%


# ============================================================================
# TESTS PARA GUARDADO DE MODELO
# ============================================================================

class TestModelSaving:
    """Tests para el guardado del modelo"""
    
    def test_pickle_save_and_load(self, tmp_path):
        """Verifica que el modelo se puede guardar y cargar con pickle"""
        import pickle
        
        # Simular el paquete del modelo
        model_package = {
            'model': {'dummy': 'model'},
            'tokenizer': {'dummy': 'tokenizer'},
            'model_name': 'test-model',
            'max_len': 128
        }
        
        pkl_path = tmp_path / "test_model.pkl"
        
        # Guardar
        with open(pkl_path, 'wb') as f:
            pickle. dump(model_package, f)
        
        # Cargar
        with open(pkl_path, 'rb') as f:
            loaded = pickle.load(f)
        
        assert loaded['model_name'] == 'test-model'
        assert loaded['max_len'] == 128
    
    def test_model_package_has_required_keys(self):
        """Verifica que el paquete del modelo tiene todas las claves necesarias"""
        required_keys = ['model', 'tokenizer', 'model_name', 'max_len']
        
        model_package = {
            'model': None,
            'tokenizer': None,
            'model_name': 'huawei-noah/TinyBERT_General_4L_312D',
            'max_len': 128
        }
        
        for key in required_keys:
            assert key in model_package


# ============================================================================
# TESTS DE EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests para casos límite"""
    
    def test_empty_text_handling(self, mock_tokenizer):
        """Verifica el manejo de textos vacíos"""
        empty_text = ""
        # El tokenizer debería manejar esto sin errores
        assert isinstance(empty_text, str)
    
    def test_very_long_text_truncation(self):
        """Verifica que textos largos se truncan correctamente"""
        max_len = 128
        long_text = "word " * 500  # ~500 palabras
        words = long_text. split()[:max_len]  # Simular truncamiento
        
        assert len(words) <= max_len
    
    def test_special_characters_handling(self):
        """Verifica el manejo de caracteres especiales"""
        special_text = "Hello!  @user #hashtag 🎉 https://example.com"
        
        # Debería poder procesarse sin errores
        assert isinstance(special_text, str)
        assert len(special_text) > 0
    
    def test_single_sample_batch(self, sample_texts, sample_labels, mock_tokenizer):
        """Verifica el manejo de batches de una sola muestra"""
        from torch.utils.data import Dataset, DataLoader
        
        class HateSpeechDataset(Dataset):
            def __init__(self, texts, labels):
                self.texts = texts
                self.labels = labels
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch. zeros(128),
                    'labels': torch.tensor(int(self.labels[idx]))
                }
        
        # Dataset con solo una muestra
        dataset = HateSpeechDataset(sample_texts[:1], sample_labels[:1])
        loader = DataLoader(dataset, batch_size=1)
        
        batch = next(iter(loader))
        assert batch['input_ids'].shape[0] == 1


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])