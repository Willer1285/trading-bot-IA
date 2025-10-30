import asyncio
import sys
import os
import io
from pathlib import Path
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

# --- Setup ---
# Load .env file
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import config
from src.ai_engine.feature_engineering import FeatureEngineer
from src.ai_engine.ai_models import EnsembleModel, SimplePatternModel, create_meta_labels

# --- Configuration ---
MODEL_OUTPUT_DIR = "models"
HISTORICAL_DATA_DIR = "historical_data"

# --- Main Training Logic ---

def prepare_training_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """Prepares the feature set (X) and the target variable (y) for meta-labeling."""
    logger.info("Preparing data for training with Meta-Labeling...")
    feature_engineer = FeatureEngineer()

    # 1. Engineer features
    df_features = feature_engineer.extract_features(df)

    # 2. Get primary model (rules) predictions
    logger.info("Generating signals from primary model (SimplePatternModel)...")
    primary_model = SimplePatternModel()
    
    # Prepare a temporary feature set for the primary model
    temp_X = feature_engineer.prepare_for_model(df_features)
    
    # Align indices before making predictions
    df_features_aligned = df_features.loc[temp_X.index]
    
    primary_predictions = pd.Series(primary_model.predict(temp_X), index=temp_X.index)

    # 3. Create meta-labels based on primary model signals
    meta_labels = create_meta_labels(df.loc[temp_X.index], primary_predictions)
    
    # 4. Combine features and labels
    df_combined = df_features_aligned.join(meta_labels)

    # 5. Clean data and select final features
    df_clean = feature_engineer.prepare_for_model(df_combined, target_column='meta_label').dropna()
    
    if df_clean.empty:
        logger.warning("No signals from primary model to create meta-labels. Training will be skipped.")
        return pd.DataFrame(), pd.Series()

    X = df_clean[feature_engineer.get_feature_importance_names()]
    y = df_clean['meta_label']
    
    # Ensure y is a 1D array
    if y.ndim > 1 and 'meta_label' in df_clean.columns:
        y = df_clean['meta_label']
    elif y.ndim > 1:
        y = y.iloc[:, 0]

    logger.info(f"Data preparation complete. Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    return X, y

def load_all_historical_data() -> pd.DataFrame:
    """Loads and combines all historical data from the local CSV files."""
    logger.info(f"Loading all historical data from '{HISTORICAL_DATA_DIR}'...")
    all_data = []
    
    for symbol_dir in os.listdir(HISTORICAL_DATA_DIR):
        symbol_path = os.path.join(HISTORICAL_DATA_DIR, symbol_dir)
        if os.path.isdir(symbol_path):
            for timeframe_file in os.listdir(symbol_path):
                if timeframe_file.endswith('.csv'):
                    file_path = os.path.join(symbol_path, timeframe_file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().replace('"', '')
                        
                        df = pd.read_csv(io.StringIO(content), sep='\t')
                        
                        # Limpiar los nombres de las columnas
                        df.columns = df.columns.str.replace(r'[<>]', '', regex=True).str.strip()
                        
                        # Se procesan las columnas para que coincidan con el formato interno.
                        if 'TIME' in df.columns and 'DATE' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], errors='coerce')
                        elif 'DATE' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['DATE'], format='%Y.%m.%d', errors='coerce')
                        else:
                            logger.warning(f"Skipping {file_path} due to missing 'DATE' or 'TIME' columns.")
                            continue

                        df.rename(columns={
                            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
                            'CLOSE': 'close', 'VOL': 'volume'
                        }, inplace=True)
                        df['symbol'] = symbol_dir
                        df['timeframe'] = timeframe_file.replace('.csv', '')
                        
                        # Asegurarse de que todas las columnas necesarias están presentes
                        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe']
                        if all(col in df.columns for col in required_cols):
                            all_data.append(df[required_cols])
                            logger.info(f"Loaded {len(df)} records from {file_path}")
                        else:
                            logger.warning(f"Skipping {file_path} due to missing columns.")
                            
                    except Exception as e:
                        logger.error(f"Failed to load or process {file_path}: {e}")

    if not all_data:
        return pd.DataFrame()

    # Se combinan todos los DataFrames en uno solo y se ordena por fecha.
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.set_index('timestamp', inplace=True)
    combined_df.sort_index(inplace=True)
    return combined_df

async def main():
    """Main function to run the training process."""
    logger.add("logs/training.log", rotation="10 MB", level="INFO")
    logger.info("=" * 30 + " Starting AI Model Training " + "=" * 30)

    # 1. Load all historical data from local files
    historical_data = load_all_historical_data()

    if historical_data.empty:
        logger.error("No historical data found or loaded. Aborting training.")
        return
    
    logger.success(f"Successfully loaded a total of {len(historical_data)} data points for all symbols and timeframes.")

    # 2. Prepare data
    X, y = prepare_training_data(historical_data)
    
    if X.empty or y.empty:
        logger.error("Data preparation resulted in empty dataframes. Aborting training.")
        return

    # 3. Create sequences for LSTM
    feature_engineer = FeatureEngineer()
    X_seq, y_seq = feature_engineer.create_sequences(X, y)

    if X_seq.shape[0] == 0:
        logger.error("Not enough data to create sequences for LSTM. Aborting.")
        return

    # 4. Train Ensemble Model with TimeSeriesSplit
    logger.info("Initializing Stacking Ensemble model for training...")
    ensemble_model = EnsembleModel()

    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    logger.info(f"Starting Time Series Cross-Validation with {n_splits} splits...")
    
    accuracies = []
    fold = 1
    for train_index, test_index in tscv.split(X):
        # Data for RF, GB
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Sequential data for LSTM - must be re-created for each fold
        X_train_seq, y_train_seq = feature_engineer.create_sequences(X_train, y_train)
        
        if X_train_seq.shape[0] == 0:
            logger.warning(f"Skipping fold {fold} due to insufficient data for sequences.")
            continue

        logger.info(f"Fold {fold}/{n_splits}: Training on {len(X_train)} samples.")
        ensemble_model.fit(X_train, y_train, X_train_seq, y_train_seq)
        
        # Evaluate on test set
        predictions = ensemble_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
        logger.info(f"Fold {fold} Accuracy: {accuracy:.4f}")
        
        fold += 1
    
    if accuracies:
        logger.success(f"Ensemble - Average Cross-Validation Accuracy: {sum(accuracies)/len(accuracies):.4f}")

    # Final training on all data
    logger.info("-" * 20 + " Final Ensemble Model Training " + "-" * 20)
    logger.info("Training Ensemble on all available data...")
    ensemble_model.fit(X, y, X_seq, y_seq)
    logger.success("Final Ensemble model trained.")

    # 5. Save all the individual models in the ensemble
    logger.info(f"Saving individual models to '{MODEL_OUTPUT_DIR}' directory...")
    ensemble_model.save_all(MODEL_OUTPUT_DIR)
    
    logger.success(f"All models in the ensemble saved successfully to '{MODEL_OUTPUT_DIR}'.")
    logger.info("=" * 30 + " AI Model Training Finished " + "=" * 31)


if __name__ == "__main__":
    asyncio.run(main())
