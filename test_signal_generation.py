import sys
import os
import io
from pathlib import Path
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# --- Configuración ---
# Añadir el directorio src a la ruta de Python
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ai_engine.feature_engineering import FeatureEngineer
from src.ai_engine.ai_models import SimplePatternModel

# --- Constantes ---
HISTORICAL_DATA_DIR = "historical_data"
# Archivo de datos específico para la prueba
TEST_SYMBOL = "GainX 400"
TEST_TIMEFRAME = "15m"
TEST_FILE_PATH = os.path.join(HISTORICAL_DATA_DIR, TEST_SYMBOL, f"{TEST_TIMEFRAME}.txt")

def load_test_data(file_path: str) -> pd.DataFrame:
    """Carga un único archivo de datos históricos para la prueba."""
    logger.info(f"Cargando datos de prueba desde '{file_path}'...")
    if not os.path.exists(file_path):
        logger.error(f"El archivo de prueba no se encuentra en: {file_path}")
        return pd.DataFrame()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().replace('"', '')
        
        df = pd.read_csv(io.StringIO(content), sep='\t')
        
        # Limpiar los nombres de las columnas
        df.columns = df.columns.str.replace(r'[<>]', '', regex=True).str.strip()
        
        # Procesar columnas de fecha y hora
        if 'TIME' in df.columns and 'DATE' in df.columns:
            df['timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], errors='coerce')
        elif 'DATE' in df.columns:
            df['timestamp'] = pd.to_datetime(df['DATE'], format='%Y.%m.%d', errors='coerce')
        else:
            logger.error("El archivo CSV debe contener la columna 'DATE' o 'DATE' y 'TIME'.")
            return pd.DataFrame()

        df.rename(columns={
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
            'CLOSE': 'close', 'VOL': 'volume'
        }, inplace=True)
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        logger.success(f"Se cargaron {len(df)} registros desde {file_path}")
        return df

    except Exception as e:
        logger.error(f"Fallo al cargar o procesar {file_path}: {e}")
        return pd.DataFrame()

def main():
    """Función principal para ejecutar la prueba de generación de señales."""
    logger.add("logs/test_signal_generation.log", rotation="10 MB", level="INFO")
    logger.info("=" * 30 + " Prueba de Generación de Señales " + "=" * 30)

    # 1. Cargar los datos de prueba
    df = load_test_data(TEST_FILE_PATH)
    if df.empty:
        logger.error("No se pudieron cargar los datos de prueba. Abortando.")
        return

    # 2. Aplicar ingeniería de características
    logger.info("Aplicando ingeniería de características...")
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.extract_features(df)
    logger.success("Ingeniería de características completada.")

    # --- Bloque de depuración extendido ---
    logger.debug(f"Número de columnas de características extraídas: {len(feature_engineer.feature_columns)}")
    if not feature_engineer.feature_columns:
        logger.warning("La lista 'feature_columns' del ingeniero de características está vacía.")
    
    # 3. Preparar datos para el modelo primario
    temp_X = feature_engineer.prepare_for_model(df_features)
    
    if temp_X.empty:
        logger.error("No hay datos suficientes después de la preparación. Abortando.")
        return

    # 4. Generar señales con el modelo primario
    logger.info("Generando señales desde SimplePatternModel...")
    primary_model = SimplePatternModel()
    primary_predictions = pd.Series(primary_model.predict(temp_X), index=temp_X.index)

    # 5. Analizar y mostrar los resultados
    logger.info("--- Análisis de Señales Generadas ---")
    signal_counts = primary_predictions.value_counts()
    
    buy_signals = signal_counts.get(2, 0)  # 2 = COMPRAR
    sell_signals = signal_counts.get(0, 0) # 0 = VENDER
    hold_signals = signal_counts.get(1, 0) # 1 = MANTENER
    total_signals = len(primary_predictions)

    logger.info(f"Total de puntos de datos analizados: {total_signals}")
    logger.info(f"Señales de COMPRA (2): {buy_signals} ({buy_signals / total_signals:.2%})")
    logger.info(f"Señales de VENTA (0): {sell_signals} ({sell_signals / total_signals:.2%})")
    logger.info(f"Señales de MANTENER (1): {hold_signals} ({hold_signals / total_signals:.2%})")

    if buy_signals == 0 and sell_signals == 0:
        logger.warning("¡Alerta! No se generaron señales de COMPRA ni de VENTA.")
        logger.warning("Revisa la lógica en `SimplePatternModel` en `src/ai_engine/ai_models.py`.")
    else:
        logger.success("Se generaron señales de compra y/o venta con éxito.")

    logger.info("=" * 30 + " Prueba Finalizada " + "=" * 30)


if __name__ == "__main__":
    main()
