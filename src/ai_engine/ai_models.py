"""
AI Models
Ensemble machine learning models for market prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from loguru import logger
import pickle
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class BaseModel:
    """Base class for all ML models"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model"""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        raise NotImplementedError

    def save(self, path: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model {self.name} saved to {path}")

    def load(self, path: str):
        """Load model from disk and validate it"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data.get('is_fitted', False)

        # Validation check
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted.")
        
        if hasattr(self.model, 'classes_') and len(self.model.classes_) < 2:
            raise ValueError(f"Model {self.name} is invalid (only knows 1 class).")

        logger.info(f"Model {self.name} loaded and validated from {path}")


class RandomForestModel(BaseModel):
    """Random Forest Classifier"""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("RandomForest")
        base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train Random Forest"""
        unique_classes = y.nunique()
        if unique_classes < 2:
            logger.error(f"Cannot train {self.name}: only {unique_classes} unique class(es) in data. Need at least 2.")
            logger.error(f"Available classes: {y.unique()}")
            self.is_fitted = False
            return

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        logger.info(f"{self.name} trained successfully on {len(X)} samples with {unique_classes} classes")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            logger.warning(f"{self.name} not fitted, returning neutral predictions")
            return np.full(len(X), 1)  # Return HOLD

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        n_samples = len(X)
        n_classes = 3  # SELL, HOLD, BUY

        if not self.is_fitted or len(self.model.classes_) < n_classes:
            return np.full((n_samples, n_classes), 1/n_classes)

        X_scaled = self.scaler.transform(X)
        
        # Ensure output has shape (n_samples, n_classes)
        probas = self.model.predict_proba(X_scaled)
        
        if probas.shape[1] < n_classes:
            # Create a full probability array and fill it
            full_probas = np.full((n_samples, n_classes), 0.0)
            for i, class_label in enumerate(self.model.classes_):
                full_probas[:, class_label] = probas[:, i]
            return full_probas
            
        return probas


class GradientBoostingModel(BaseModel):
    """Gradient Boosting Classifier (XGBoost alternative)"""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        super().__init__("GradientBoosting")
        base_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            random_state=42
        )
        self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train Gradient Boosting"""
        unique_classes = y.nunique()
        if unique_classes < 2:
            logger.error(f"Cannot train {self.name}: only {unique_classes} unique class(es) in data. Need at least 2.")
            logger.error(f"Available classes: {y.unique()}")
            self.is_fitted = False
            return

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        logger.info(f"{self.name} trained successfully on {len(X)} samples with {unique_classes} classes")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            return np.full(len(X), 1)  # Return HOLD

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        n_samples = len(X)
        n_classes = 3  # SELL, HOLD, BUY

        if not self.is_fitted or len(self.model.classes_) < n_classes:
            return np.full((n_samples, n_classes), 1/n_classes)

        X_scaled = self.scaler.transform(X)
        
        # Ensure output has shape (n_samples, n_classes)
        probas = self.model.predict_proba(X_scaled)
        
        if probas.shape[1] < n_classes:
            # Create a full probability array and fill it
            full_probas = np.full((n_samples, n_classes), 0.0)
            for i, class_label in enumerate(self.model.classes_):
                full_probas[:, class_label] = probas[:, i]
            return full_probas
            
        return probas


class SimplePatternModel(BaseModel):
    """
    Modelo simple basado en reglas y patrones de indicadores técnicos.
    Este modelo no requiere entrenamiento (`fit`) y genera señales de compra/venta
    basadas en una puntuación calculada a partir de RSI, MACD y una tendencia simple.
    """

    def __init__(self):
        """Inicializa el modelo. Es 'fitted' por defecto porque no necesita entrenamiento."""
        super().__init__("PatternBased")
        self.is_fitted = True  # Basado en reglas, siempre está listo.

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """No se necesita entrenamiento para un modelo basado en reglas."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice señales de trading (COMPRAR, VENDER, MANTENER) basadas en indicadores técnicos.
        
        La lógica es la siguiente:
        1.  Crea una "memoria" para el RSI para detectar condiciones de sobrecompra/sobreventa recientes.
        2.  Calcula una puntuación (`score`) para cada punto de datos:
            - RSI recientemente sobrevendido: +2
            - RSI recientemente sobrecomprado: -2
            - MACD alcista: +1
            - MACD bajista: -1
            - Tendencia alcista: +1
            - Tendencia bajista: -1
        3.  Clasifica la señal basada en la puntuación final:
            - Puntuación >= 1: COMPRAR (2)
            - Puntuación <= -1: VENDER (0)
            - De lo contrario: MANTENER (1)
        """
        
        X_with_memory = X.copy()
        
        # Se crean características con "memoria" de las condiciones recientes del RSI.
        # Se comprueba si el RSI cruzó el umbral en algún momento de las últimas 'window' velas.
        window = 20
        if 'rsi_14' in X.columns:
            X_with_memory['rsi_recently_oversold'] = X['rsi_14'].rolling(window=window).apply(lambda x: (x < 30).any(), raw=True).fillna(0).astype(bool)
            X_with_memory['rsi_recently_overbought'] = X['rsi_14'].rolling(window=window).apply(lambda x: (x > 70).any(), raw=True).fillna(0).astype(bool)
        else:
            X_with_memory['rsi_recently_oversold'] = False
            X_with_memory['rsi_recently_overbought'] = False

        predictions = []

        for _, row in X_with_memory.iterrows():
            score = 0

            # Señales del RSI con memoria
            if row['rsi_recently_oversold']:
                score += 2
            if row['rsi_recently_overbought']:
                score -= 2

            # Señales del MACD
            if 'macd_diff' in row:
                if row['macd_diff'] > 0:
                    score += 1  # Alcista
                else:
                    score -= 1  # Bajista

            # Señales de tendencia
            if 'trend_20' in row:
                score += row['trend_20']  # Esto ya es 1 para alcista, -1 para bajista

            # Clasificar la señal.
            # Se redujeron los umbrales de 1.5 a 1 para generar más señales candidatas
            # para el modelo secundario.
            if score >= 1:
                predictions.append(2)  # COMPRAR
            elif score <= -1:
                predictions.append(0)  # VENDER
            else:
                predictions.append(1)  # MANTENER

        return np.array(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probabilities from predictions"""
        predictions = self.predict(X)
        probas = np.zeros((len(predictions), 3))

        for i, pred in enumerate(predictions):
            if pred == 2:  # BUY
                probas[i] = [0.1, 0.2, 0.7]
            elif pred == 0:  # SELL
                probas[i] = [0.7, 0.2, 0.1]
            else:  # HOLD
                probas[i] = [0.2, 0.6, 0.2]

        return probas


class EnsembleModel:
    """
    Ensemble of multiple models using Stacking.
    A meta-model is trained on the predictions of the base models.
    """

    def __init__(self):
        """Initialize ensemble with base models and a meta-model."""
        self.base_models = {
            'random_forest': RandomForestModel(n_estimators=100, max_depth=10),
            'gradient_boosting': GradientBoostingModel(n_estimators=100, learning_rate=0.1),
            'pattern_based': SimplePatternModel(),
            'lstm': None  # LSTM model will be created at fit time
        }
        # Meta-model learns from the outputs of base models
        self.meta_model = LogisticRegression()
        self.is_fitted = False
        logger.info("Initialized Stacking Ensemble with Logistic Regression as meta-model.")

    def _get_base_model_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get predictions from all base models to use as features for the meta-model."""
        base_predictions = {}
        
        # This is a workaround to avoid circular imports and re-instantiate a temporary engineer
        from .feature_engineering import FeatureEngineer
        feature_engineer = FeatureEngineer()

        for name, model in self.base_models.items():
            if model is None:
                # If a model (like LSTM) is optional and not loaded, we must add placeholder columns
                # that the meta-model expects.
                if name == 'lstm':
                    logger.warning(f"LSTM model not loaded. Using neutral probabilities as placeholders.")
                    base_predictions[f"{name}_proba_0"] = 0.5
                    base_predictions[f"{name}_proba_1"] = 0.5
                else:
                    logger.warning(f"Model {name} is not available, skipping its prediction.")
                continue

            if name == 'lstm':
                # LSTM requires sequence data.
                dummy_y = pd.Series(np.zeros(len(X)), index=X.index)
                X_seq, _ = feature_engineer.create_sequences(X, dummy_y, default_sequence_length=model.sequence_length)
                
                if X_seq.shape[0] > 0:
                    probas = model.predict_proba(X_seq)
                    # Align predictions with the original index
                    pred_index = X.index[model.sequence_length : model.sequence_length + len(probas)]
                    for i in range(probas.shape[1]):
                        base_predictions[f"{name}_proba_{i}"] = pd.Series(probas[:, i], index=pred_index)
                else:
                    # If not enough data for a sequence, provide neutral placeholders
                    logger.warning(f"Not enough data for LSTM sequence. Using neutral probabilities as placeholders.")
                    base_predictions[f"{name}_proba_0"] = 0.5
                    base_predictions[f"{name}_proba_1"] = 0.5

            elif name != 'pattern_based':
                # Tabular models
                probas = model.predict_proba(X)
                for i in range(probas.shape[1]):
                    base_predictions[f"{name}_proba_{i}"] = pd.Series(probas[:, i], index=X.index)
            else:
                # Rule-based model
                base_predictions[name] = pd.Series(model.predict(X), index=X.index)

        # Create a DataFrame and align all predictions.
        return pd.DataFrame(base_predictions, index=X.index)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_seq: np.ndarray, y_seq: np.ndarray):
        """Train all base models and the meta-model."""
        # Create LSTM model on the fly if it doesn't exist
        if self.base_models['lstm'] is None:
            logger.info("Creating LSTM model...")
            # Infer input_dim from the shape of the sequence data
            input_dim = X_seq.shape[2]
            self.base_models['lstm'] = LSTMModel(input_dim=input_dim)

        logger.info("Training base models...")
        for name, model in self.base_models.items():
            try:
                logger.info(f"Training {name}...")
                if name == 'lstm':
                    model.fit(X_seq, y_seq)
                elif name != 'pattern_based':
                    model.fit(X, y)
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        logger.info("Generating base model predictions for meta-model training...")
        meta_features = self._get_base_model_predictions(X)
        
        # Combine features and target, then drop rows with NaNs (from LSTM sequence creation)
        combined_data = meta_features.join(y.rename('target'))
        combined_data.dropna(inplace=True)
        
        clean_meta_features = combined_data.drop('target', axis=1)
        clean_y = combined_data['target']

        if clean_meta_features.empty:
            logger.error("Meta-features are empty after handling LSTM predictions. Cannot train meta-model.")
            self.is_fitted = False
            return

        logger.info("Training meta-model...")
        self.meta_model.fit(clean_meta_features, clean_y)
        self.is_fitted = True
        logger.success("Stacking Ensemble trained successfully.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction using the meta-model."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble model must be fitted before making predictions.")
        
        meta_features = self._get_base_model_predictions(X)
        # Forward-fill any NaNs that may result from the LSTM sequence alignment
        meta_features.ffill(inplace=True)
        meta_features.bfill(inplace=True) # Back-fill for any remaining at the start
        
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted ensemble probabilities from the meta-model."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble model must be fitted before making predictions.")
            
        meta_features = self._get_base_model_predictions(X)
        # Forward-fill any NaNs that may result from the LSTM sequence alignment
        meta_features.ffill(inplace=True)
        meta_features.bfill(inplace=True) # Back-fill for any remaining at the start

        return self.meta_model.predict_proba(meta_features)

    def predict_individual(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model"""
        predictions = {}
        for name, model in self.base_models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                logger.error(f"Error getting prediction from {name}: {e}")
                predictions[name] = np.array([1] * len(X))  # Default to HOLD
        return predictions

    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get confidence scores for predictions

        Returns:
            Array of confidence scores (0-1)
        """
        probas = self.predict_proba(X)
        # Confidence is the maximum probability
        return np.max(probas, axis=1)

    def save_all(self, directory: str):
        """Save all models, including the meta-model."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save base models
        for name, model in self.base_models.items():
            if name != 'pattern_based' and model is not None:
                model_path = path / f"{name}.pkl"
                model.save(str(model_path))
        
        # Save the meta-model
        if self.is_fitted:
            meta_model_path = path / "meta_model.pkl"
            with open(meta_model_path, 'wb') as f:
                pickle.dump(self.meta_model, f)
            logger.info(f"Meta-model saved to {meta_model_path}")

    def load_all(self, directory: str):
        """Load all models, including the meta-model, and validate."""
        path = Path(directory)
        
        # Load base models
        if self.base_models['lstm'] is None:
            self.base_models['lstm'] = LSTMModel()

        models_loaded = 0
        models_expected = 0
        for name, model in self.base_models.items():
            if name != 'pattern_based':
                models_expected += 1
                model_path = path / f"{name}.pkl"
                if model_path.exists():
                    try:
                        model.load(str(model_path))
                        models_loaded += 1
                        logger.info(f"Successfully loaded model: {name}")
                    except Exception as e:
                        logger.error(f"Failed to load {name}: {e}")
                else:
                    logger.warning(f"Model file not found: {model_path}")
                    if name == 'lstm':
                        self.base_models['lstm'] = None
                        models_expected -= 1

        if models_loaded == 0:
            raise FileNotFoundError(f"No base model files found in {directory}")

        # Load the meta-model
        meta_model_path = path / "meta_model.pkl"
        if meta_model_path.exists():
            try:
                with open(meta_model_path, 'rb') as f:
                    self.meta_model = pickle.load(f)
                logger.info(f"Meta-model loaded successfully from {meta_model_path}")
                self.is_fitted = True  # Mark ensemble as ready
            except Exception as e:
                logger.error(f"Failed to load meta-model: {e}")
                self.is_fitted = False
        else:
            logger.error("Meta-model not found. The ensemble cannot make predictions.")
            self.is_fitted = False

        if models_loaded < models_expected:
            logger.warning(f"Only {models_loaded}/{models_expected} base models loaded. Performance may be degraded.")


def calculate_optimal_threshold(df: pd.DataFrame, forward_window: int = 5) -> float:
    """
    Calculate optimal threshold based on price volatility

    Args:
        df: DataFrame with price data
        forward_window: Number of periods to look forward

    Returns:
        Optimal threshold value
    """
    # Calculate returns
    returns = df['close'].pct_change().dropna()

    # Calculate volatility (standard deviation of returns)
    volatility = returns.std()

    # Calculate average price movement over forward_window periods
    rolling_changes = []
    for i in range(len(df) - forward_window):
        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+forward_window+1]
        max_change = abs((future_prices.max() - current_price) / current_price)
        min_change = abs((current_price - future_prices.min()) / current_price)
        rolling_changes.append(max(max_change, min_change))

    avg_movement = np.mean(rolling_changes) if rolling_changes else 0.02

    # Use 30th percentile of movements as threshold
    # This ensures we capture significant moves but not too conservative
    threshold = np.percentile(rolling_changes, 30) if rolling_changes else 0.01

    # Ensure threshold is reasonable (between 0.1% and 10%)
    threshold = max(0.001, min(threshold, 0.10))

    logger.info(f"Calculated optimal threshold: {threshold:.4f} (volatility: {volatility:.4f}, avg_movement: {avg_movement:.4f})")

    return threshold


def create_meta_labels(
    df: pd.DataFrame,
    primary_model_predictions: pd.Series,
    forward_window: int = 15
) -> pd.Series:
    """
    Create meta-labels for a secondary model (Meta-Labeling) using a dynamic threshold.

    The secondary model decides whether to take a trade based on the primary model's signal.
    The label is 1 if the trade is profitable (hits profit-take), 0 otherwise (hits stop-loss or expires).

    Args:
        df: DataFrame with OHLCV data.
        primary_model_predictions: Series of predictions from the primary model (0=SELL, 1=HOLD, 2=BUY).
        forward_window: Max number of periods to hold a trade.

    Returns:
        A Series of meta-labels (1 for profitable trade, 0 for unprofitable/expired trade).
    """
    # Dynamically calculate thresholds based on market volatility
    threshold = calculate_optimal_threshold(df, forward_window=forward_window)
    profit_take_pct = threshold * 1.5  # Target 1.5x the typical movement
    stop_loss_pct = threshold * 1.0   # Stop loss at 1x the typical movement
    logger.info("Creating meta-labels for secondary model...")
    labels = pd.Series(np.nan, index=df.index)
    
    profit_take_count = 0
    stop_loss_count = 0

    for i in range(len(df) - forward_window):
        signal = primary_model_predictions.iloc[i]
        
        # Only generate labels for BUY or SELL signals
        if signal == 1:  # HOLD
            continue

        entry_price = df['close'].iloc[i]
        
        # Define profit take and stop loss levels
        if signal == 2:  # BUY signal
            profit_target = entry_price * (1 + profit_take_pct)
            stop_loss = entry_price * (1 - stop_loss_pct)
        else:  # SELL signal
            profit_target = entry_price * (1 - profit_take_pct)
            stop_loss = entry_price * (1 + stop_loss_pct)

        # Look into the future to see what happens first
        outcome = 0  # Default to loss/expired
        for j in range(1, forward_window + 1):
            future_high = df['high'].iloc[i + j]
            future_low = df['low'].iloc[i + j]

            if signal == 2:  # BUY
                if future_high >= profit_target:
                    outcome = 1  # Profitable
                    profit_take_count += 1
                    break
                if future_low <= stop_loss:
                    outcome = 0  # Loss
                    stop_loss_count += 1
                    break
            else:  # SELL
                if future_low <= profit_target:
                    outcome = 1  # Profitable
                    profit_take_count += 1
                    break
                if future_high >= stop_loss:
                    outcome = 0  # Loss
                    stop_loss_count += 1
                    break
        
        labels.iloc[i] = outcome

    total_signals = profit_take_count + stop_loss_count
    if total_signals > 0:
        logger.info(f"Meta-Label distribution - Profitable: {profit_take_count} ({profit_take_count/total_signals*100:.1f}%), "
                    f"Unprofitable: {stop_loss_count} ({stop_loss_count/total_signals*100:.1f}%)")
    else:
        logger.warning("No BUY/SELL signals found to create meta-labels.")

    return labels.rename('meta_label')


class LSTMModel(BaseModel):
    """LSTM model for sequence classification."""

    def __init__(self, sequence_length: int = 50, input_dim: int = 50):
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        # Defer model building until fit or load
        self.model = None

    def _build_model(self):
        """Build the Keras LSTM model."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, self.input_dim)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the LSTM model."""
        if self.model is None:
            self._build_model()
            
        # X and y are expected to be numpy arrays of sequences
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Data for LSTM doesn't need the same scaling as tabular models, but we'll scale it anyway for consistency
        # Reshape for scaling
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape((n_samples * n_timesteps, n_features))
        self.scaler.fit(X_reshaped)
        X_scaled_reshaped = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape((n_samples, n_timesteps, n_features))

        self.model.fit(
            X_scaled, y,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        self.is_fitted = True
        logger.info(f"{self.name} trained successfully.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (0 or 1)."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before making predictions.")
        
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape((n_samples * n_timesteps, n_features))
        X_scaled_reshaped = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape((n_samples, n_timesteps, n_features))

        return (self.model.predict(X_scaled) > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before making predictions.")
        
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape((n_samples * n_timesteps, n_features))
        X_scaled_reshaped = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape((n_samples, n_timesteps, n_features))
        
        proba_positive = self.model.predict(X_scaled)
        proba_negative = 1 - proba_positive
        return np.hstack([proba_negative, proba_positive])

    def save(self, path: str):
        """Save Keras model and scaler separately."""
        if self.model is None:
            logger.warning("Attempted to save an un-built LSTM model.")
            return
            
        model_path = path.replace('.pkl', '.keras')
        scaler_path = path.replace('.pkl', '_scaler.pkl')
        
        self.model.save(model_path)
        
        scaler_data = {'scaler': self.scaler, 'is_fitted': self.is_fitted}
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_data, f)

        # Create a dummy .pkl file so that EnsembleModel.load_all can find it
        Path(path).touch()
            
        logger.info(f"LSTM model saved to {model_path}, {scaler_path} and marker file {path}")

    def load(self, path: str):
        """Load Keras model and scaler."""
        model_path = path.replace('.pkl', '.keras')
        scaler_path = path.replace('.pkl', '_scaler.pkl')

        if not Path(model_path).exists() or not Path(scaler_path).exists():
            raise FileNotFoundError(f"Model files not found: {model_path} or {scaler_path}")

        self.model = load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.scaler = scaler_data['scaler']
        self.is_fitted = scaler_data['is_fitted']
        
        logger.info(f"LSTM model loaded from {model_path} and {scaler_path}")
