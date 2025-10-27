import asyncio
import sys
import os
from pathlib import Path
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
import joblib

# --- Setup ---
# Load .env file
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_collector.mt5_connector import MT5Connector
from src.ai_engine.feature_engineering import FeatureEngineer
from src.ai_engine.ai_models import RandomForestModel, GradientBoostingModel

# --- Configuration ---
# Symbols and timeframe to use for training. Using one symbol is usually enough.
TRAIN_SYMBOL = os.getenv("TRAIN_SYMBOL", "PainX 400") 
TRAIN_TIMEFRAME = os.getenv("TRAIN_TIMEFRAME", "15m")
DATA_POINTS = int(os.getenv("TRAIN_DATA_POINTS", 50000)) # Number of candles to train on
MODEL_OUTPUT_DIR = "models"

# --- Main Training Logic ---

def prepare_training_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """Prepares the feature set (X) and the target variable (y) for training."""
    logger.info("Preparing data for training...")
    
    # 1. Create the target variable 'y'
    # We want to predict if the price will go up or down in the near future.
    # Let's define "future" as 10 candles from now.
    future_price = df['close'].shift(-10)
    
    # Create target: 1 for price up, 0 for price down/same
    df['target'] = (future_price > df['close']).astype(int)
    
    # 2. Engineer features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.extract_features(df)
    
    # 3. Clean data and select features
    # We drop the last 10 rows because they don't have a future price to compare against (NaNs)
    df_clean = feature_engineer.prepare_for_model(df_features, target_column='target').dropna()
    
    X = df_clean[feature_engineer.get_feature_importance_names()]
    y = df_clean['target']
    
    # Ensure y is a 1D array
    if y.ndim > 1:
        y = y.iloc[:, 0]
    
    logger.info(f"Data preparation complete. Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    return X, y

async def main():
    """Main function to run the training process."""
    logger.add("logs/training.log", rotation="10 MB", level="INFO")
    logger.info("=" * 30 + " Starting AI Model Training " + "=" * 30)

    # 1. Connect to MT5 and get data
    logger.info("Connecting to MT5...")
    mt5_connector = MT5Connector(
        login=int(os.getenv('MT5_LOGIN')),
        password=os.getenv('MT5_PASSWORD'),
        server=os.getenv('MT5_SERVER')
    )
    if not mt5_connector.is_connected:
        logger.error("Failed to connect to MT5. Aborting training.")
        return

    logger.info(f"Fetching {DATA_POINTS} data points for {TRAIN_SYMBOL} on {TRAIN_TIMEFRAME} timeframe...")
    historical_data = await mt5_connector.fetch_ohlcv(TRAIN_SYMBOL, TRAIN_TIMEFRAME, DATA_POINTS)
    mt5_connector.shutdown()

    if historical_data.empty:
        logger.error("Failed to fetch historical data. Aborting training.")
        return
    
    logger.success(f"Successfully fetched {len(historical_data)} data points.")

    # 2. Prepare data
    X, y = prepare_training_data(historical_data)
    
    if X.empty or y.empty:
        logger.error("Data preparation resulted in empty dataframes. Aborting training.")
        return

    # 3. Train models
    logger.info("Initializing AI models for training...")
    rf_model = RandomForestModel()
    gb_model = GradientBoostingModel()
    
    logger.info("Training Random Forest model...")
    rf_model.model.fit(X, y)
    logger.success("Random Forest model trained.")
    
    logger.info("Training Gradient Boosting model...")
    gb_model.model.fit(X, y)
    logger.success("Gradient Boosting model trained.")

    # 4. Save models
    logger.info(f"Saving models to '{MODEL_OUTPUT_DIR}' directory...")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    rf_path = os.path.join(MODEL_OUTPUT_DIR, "random_forest.pkl")
    gb_path = os.path.join(MODEL_OUTPUT_DIR, "gradient_boosting.pkl")
    
    joblib.dump(rf_model.model, rf_path)
    joblib.dump(gb_model.model, gb_path)
    
    logger.success(f"Models saved successfully: {rf_path}, {gb_path}")
    logger.info("=" * 30 + " AI Model Training Finished " + "=" * 31)


if __name__ == "__main__":
    asyncio.run(main())
