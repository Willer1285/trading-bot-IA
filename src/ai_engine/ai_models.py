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
from loguru import logger
import pickle
from pathlib import Path


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
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train Random Forest"""
        if y.nunique() < 2:
            logger.warning(f"Skipping training for {self.name}: not enough classes in data.")
            self.is_fitted = False
            return
            
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        logger.info(f"{self.name} trained on {len(X)} samples")

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
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            random_state=42
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train Gradient Boosting"""
        if y.nunique() < 2:
            logger.warning(f"Skipping training for {self.name}: not enough classes in data.")
            self.is_fitted = False
            return

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        logger.info(f"{self.name} trained on {len(X)} samples")

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
    """Simple rule-based pattern model"""

    def __init__(self):
        super().__init__("PatternBased")
        self.is_fitted = True  # Rule-based, always ready

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """No training needed for rule-based model"""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict based on technical indicators"""
        predictions = []

        for _, row in X.iterrows():
            score = 0

            # RSI signals
            if 'rsi_14' in row:
                if row['rsi_14'] < 30:
                    score += 2  # Oversold - Buy
                elif row['rsi_14'] > 70:
                    score -= 2  # Overbought - Sell

            # MACD signals
            if 'macd_diff' in row:
                if row['macd_diff'] > 0:
                    score += 1  # Bullish
                else:
                    score -= 1  # Bearish

            # Trend signals
            if 'sma_25' in row and 'close' in row:
                if row['close'] > row['sma_25']:
                    score += 1
                else:
                    score -= 1

            # Classify
            if score >= 2:
                predictions.append(2)  # BUY
            elif score <= -2:
                predictions.append(0)  # SELL
            else:
                predictions.append(1)  # HOLD

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
    """Ensemble of multiple models with weighted voting"""

    def __init__(self, model_weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble

        Args:
            model_weights: Dictionary mapping model name to weight
        """
        self.models = {
            'random_forest': RandomForestModel(n_estimators=100, max_depth=10),
            'gradient_boosting': GradientBoostingModel(n_estimators=100, learning_rate=0.1),
            'pattern_based': SimplePatternModel()
        }

        self.weights = model_weights or {
            'random_forest': 0.35,
            'gradient_boosting': 0.35,
            'pattern_based': 0.30
        }

        # Ensure weights sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}

        logger.info(f"Initialized ensemble with weights: {self.weights}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train all models in ensemble"""
        for name, model in self.models.items():
            if name != 'pattern_based':  # Pattern model doesn't need training
                try:
                    logger.info(f"Training {name}...")
                    model.fit(X, y)
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction"""
        # Get probabilities from all models
        probas = self.predict_proba(X)

        # Return class with highest probability
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted ensemble probabilities"""
        all_probas = []

        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X)
                weight = self.weights.get(name, 0)
                all_probas.append(proba * weight)
            except Exception as e:
                logger.error(f"Error getting predictions from {name}: {e}")

        if not all_probas:
            # Return neutral probabilities
            return np.full((len(X), 3), 1/3)

        # Weighted average
        ensemble_proba = np.sum(all_probas, axis=0)

        return ensemble_proba

    def predict_individual(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model"""
        predictions = {}
        for name, model in self.models.items():
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
        """Save all models"""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            if name != 'pattern_based':  # Don't save rule-based model
                model_path = path / f"{name}.pkl"
                model.save(str(model_path))

    def load_all(self, directory: str):
        """Load all models"""
        path = Path(directory)

        for name, model in self.models.items():
            if name != 'pattern_based':
                model_path = path / f"{name}.pkl"
                if model_path.exists():
                    model.load(str(model_path))
                else:
                    logger.warning(f"Model file not found: {model_path}")


def create_labels(df: pd.DataFrame, forward_window: int = 5, threshold: float = 0.02) -> pd.Series:
    """
    Create labels for supervised learning

    Args:
        df: DataFrame with price data
        forward_window: Number of periods to look forward
        threshold: Minimum price change to trigger signal (2% default)

    Returns:
        Series with labels (0=SELL, 1=HOLD, 2=BUY)
    """
    labels = []

    for i in range(len(df) - forward_window):
        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+forward_window+1]

        max_future = future_prices.max()
        min_future = future_prices.min()

        price_change_up = (max_future - current_price) / current_price
        price_change_down = (current_price - min_future) / current_price

        if price_change_up > threshold and price_change_up > price_change_down:
            labels.append(2)  # BUY
        elif price_change_down > threshold and price_change_down > price_change_up:
            labels.append(0)  # SELL
        else:
            labels.append(1)  # HOLD

    # Fill remaining with HOLD
    labels.extend([1] * forward_window)

    return pd.Series(labels, index=df.index, name='label')
