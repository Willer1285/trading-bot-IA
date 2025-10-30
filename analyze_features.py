import asyncio
import sys
import os
from pathlib import Path
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
import joblib
import shap
import matplotlib.pyplot as plt

# --- Setup ---
# Load .env file
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_collector.mt5_connector import MT5Connector
from src.ai_engine.feature_engineering import FeatureEngineer
from src.ai_engine.ai_models import SimplePatternModel, create_meta_labels
from train_models import prepare_training_data

# --- Configuration ---
MODEL_INPUT_DIR = "models"
CHART_OUTPUT_DIR = "docs/feature_importance"
DATA_POINTS = int(os.getenv("TRAIN_DATA_POINTS", 10000)) # Use a smaller dataset for faster analysis
TRAIN_SYMBOL = os.getenv("TRAIN_SYMBOL", "PainX 400")
TRAIN_TIMEFRAME = os.getenv("TRAIN_TIMEFRAME", "15m")

async def main():
    """Main function to run the feature analysis."""
    logger.add("logs/feature_analysis.log", rotation="10 MB", level="INFO")
    logger.info("=" * 30 + " Starting Feature Importance Analysis " + "=" * 30)

    # 1. Load Ensemble Model
    logger.info(f"Loading ensemble model from '{MODEL_INPUT_DIR}' directory...")
    os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)
    
    ensemble_path = os.path.join(MODEL_INPUT_DIR, "ensemble_model.pkl")

    if not os.path.exists(ensemble_path):
        logger.error("Ensemble model file not found. Please run train_models.py first.")
        return

    ensemble_model = joblib.load(ensemble_path)
    logger.success("Ensemble model loaded successfully.")

    # Extract base models for analysis
    rf_model = ensemble_model.base_models['random_forest'].model
    gb_model = ensemble_model.base_models['gradient_boosting'].model

    # 2. Get Data for Analysis
    logger.info("Connecting to MT5 to fetch data for analysis...")
    mt5_connector = MT5Connector(
        login=int(os.getenv('MT5_LOGIN')),
        password=os.getenv('MT5_PASSWORD'),
        server=os.getenv('MT5_SERVER')
    )
    if not mt5_connector.is_connected:
        logger.error("Failed to connect to MT5. Aborting analysis.")
        return

    historical_data = await mt5_connector.fetch_ohlcv(TRAIN_SYMBOL, TRAIN_TIMEFRAME, DATA_POINTS)
    mt5_connector.shutdown()

    if historical_data.empty:
        logger.error("Failed to fetch historical data. Aborting analysis.")
        return
    
    X, y = prepare_training_data(historical_data)

    if X.empty:
        logger.error("No data available for analysis after preparation.")
        return
        
    # Subsample for faster SHAP computation
    X_sample = X.sample(n=min(1000, len(X)), random_state=42)

    # 3. Analyze Models with SHAP
    models_to_analyze = {
        "RandomForest": rf_model,
        "GradientBoosting": gb_model
    }

    for name, model in models_to_analyze.items():
        logger.info(f"Calculating SHAP values for {name}...")
        
        # SHAP explainer requires the model's prediction function
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # For classifiers, shap_values can be a list of arrays (one for each class)
        # We'll use the values for the "positive" class (1) for the summary plot
        class_index = 1 if isinstance(shap_values, list) and len(shap_values) > 1 else 0

        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values[class_index] if isinstance(shap_values, list) else shap_values, X_sample, show=False)
        plt.title(f"SHAP Feature Importance for {name}")
        plt.tight_layout()
        summary_path = os.path.join(CHART_OUTPUT_DIR, f"{name}_summary_plot.png")
        plt.savefig(summary_path)
        plt.close()
        logger.success(f"Saved SHAP summary plot to {summary_path}")

    logger.info("=" * 30 + " Feature Importance Analysis Finished " + "=" * 31)


if __name__ == "__main__":
    asyncio.run(main())
