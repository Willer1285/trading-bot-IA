# 📚 API Documentation

## Componentes Principales

### 1. ExchangeConnector

Conecta con exchanges de criptomonedas.

```python
from data_collector.exchange_connector import ExchangeConnector

# Inicializar
exchange = ExchangeConnector(
    exchange_name='binance',
    api_key='your_api_key',
    api_secret='your_secret'
)

# Obtener datos OHLCV
df = await exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)

# Obtener ticker
ticker = await exchange.fetch_ticker('BTC/USDT')
```

### 2. MarketDataManager

Gestiona recopilación de datos en tiempo real.

```python
from data_collector.market_data_manager import MarketDataManager

# Inicializar
manager = MarketDataManager(
    exchange=exchange,
    symbols=['BTC/USDT', 'ETH/USDT'],
    timeframes=['1m', '5m', '1h'],
    update_interval=60
)

# Iniciar recopilación
await manager.start()

# Obtener datos más recientes
latest = manager.get_latest_data('BTC/USDT', '1h', limit=50)

# Obtener precio actual
price = manager.get_current_price('BTC/USDT')
```

### 3. MarketAnalyzer

Motor de análisis de IA.

```python
from ai_engine.market_analyzer import MarketAnalyzer

# Inicializar
analyzer = MarketAnalyzer(enable_training=False)

# Analizar mercado
analysis = analyzer.analyze(df, 'BTC/USDT', '1h')

# Acceder a resultados
print(f"Signal: {analysis.signal}")  # BUY, SELL, HOLD
print(f"Confidence: {analysis.confidence}")  # 0-1
print(f"Indicators: {analysis.indicators}")
```

### 4. SignalGenerator

Genera señales de trading.

```python
from signal_generator.signal_generator import SignalGenerator

# Inicializar
signal_gen = SignalGenerator(
    analyzer=analyzer,
    signal_filter=signal_filter,
    risk_manager=risk_manager,
    min_confidence=0.75,
    min_strength=80
)

# Generar señal
signal = signal_gen.generate_signal(
    symbol='BTC/USDT',
    multi_tf_analyses=analyses,
    current_price=42000.0
)

# Acceder a señal
if signal:
    print(f"Type: {signal.signal_type}")
    print(f"Entry: {signal.entry_price}")
    print(f"Stop Loss: {signal.stop_loss}")
    print(f"Take Profits: {signal.take_profit_levels}")
```

### 5. TelegramBot

Bot de Telegram.

```python
from telegram_bot.telegram_bot import TelegramBot

# Inicializar
bot = TelegramBot(
    bot_token='your_token',
    channel_id='your_channel_id'
)

# Enviar señal
await bot.send_signal(signal, market_data)

# Enviar mensaje
await bot.send_message("Hello from bot!")

# Enviar resumen diario
await bot.send_daily_summary(summary)
```

## Estructuras de Datos

### TradingSignal

```python
@dataclass
class TradingSignal:
    signal_id: str              # ID único
    symbol: str                 # Par de trading
    signal_type: str            # 'BUY' o 'SELL'
    entry_price: float          # Precio de entrada
    stop_loss: float            # Stop loss
    take_profit_levels: List[float]  # [TP1, TP2, TP3]
    confidence: float           # 0-1
    strength: int               # 0-100
    timeframe: str              # Timeframe
    timestamp: datetime         # Timestamp UTC
    analysis: Dict              # Análisis completo
    risk_reward_ratio: float    # Ratio R/R
    reason: str                 # Razón de la señal
```

### MarketAnalysis

```python
class MarketAnalysis:
    symbol: str                 # Símbolo
    timeframe: str              # Timeframe
    prediction: int             # 0=SELL, 1=HOLD, 2=BUY
    confidence: float           # Confianza 0-1
    probabilities: Dict         # {'sell': 0.1, 'hold': 0.2, 'buy': 0.7}
    indicators: Dict            # Valores de indicadores
    patterns: Dict              # Patrones detectados
    support_resistance: Dict    # Niveles S/R
    timestamp: datetime         # Timestamp
```

## Personalización

### Crear Indicador Personalizado

```python
# En src/ai_engine/technical_indicators.py

@staticmethod
def custom_indicator(df: pd.DataFrame) -> pd.Series:
    """Tu indicador personalizado"""
    # Implementar lógica
    return indicator_values

# Añadir en calculate_all()
df['my_indicator'] = TechnicalIndicators.custom_indicator(df)
```

### Crear Filtro Personalizado

```python
# En src/signal_generator/signal_filter.py

def custom_filter(self, symbol: str, signal_type: str) -> bool:
    """Filtro personalizado"""
    # Tu lógica
    if condicion:
        return False  # Rechazar señal
    return True  # Aceptar señal

# Usar en should_trade()
if not self.custom_filter(symbol, signal_type):
    return False
```

### Ajustar Modelo Ensemble

```python
# En config.yaml

ai_model:
  ensemble:
    models:
      - type: "random_forest"
        weight: 0.30      # Ajustar peso
      - type: "xgboost"
        weight: 0.30
      - type: "lstm"
        weight: 0.25
      - type: "transformer"
        weight: 0.15
```

## Eventos y Callbacks

### Hook de Pre-Señal

```python
def on_signal_generated(signal: TradingSignal):
    """Se ejecuta antes de enviar señal"""
    # Logging personalizado
    logger.info(f"New signal: {signal.symbol}")

    # Validación adicional
    if signal.confidence < 0.8:
        return False  # Rechazar

    return True  # Aceptar
```

### Hook de Post-Señal

```python
def after_signal_sent(signal: TradingSignal, success: bool):
    """Se ejecuta después de enviar señal"""
    if success:
        # Guardar en base de datos
        save_to_db(signal)
```

## Testing

### Unit Tests

```python
import pytest
from ai_engine.market_analyzer import MarketAnalyzer

def test_analyzer():
    analyzer = MarketAnalyzer()

    # Test con datos mock
    df = create_mock_data()
    analysis = analyzer.analyze(df, 'BTC/USDT', '1h')

    assert analysis is not None
    assert analysis.signal in ['BUY', 'SELL', 'HOLD']
    assert 0 <= analysis.confidence <= 1
```

### Integration Tests

```bash
# Ejecutar tests
pytest tests/

# Con cobertura
pytest --cov=src tests/
```

## Logging

### Niveles de Log

```python
from loguru import logger

logger.debug("Mensaje de depuración")
logger.info("Información general")
logger.warning("Advertencia")
logger.error("Error recuperable")
logger.critical("Error crítico")
```

### Configurar Logging

```python
from utils.logger import setup_logger

setup_logger(
    log_level="DEBUG",  # DEBUG, INFO, WARNING, ERROR
    log_file="logs/bot.log",
    rotation="1 day",
    retention="30 days"
)
```

## Performance

### Optimizaciones

```python
# Reducir timeframes para análisis más rápido
TIMEFRAMES=15m,1h,4h

# Reducir pares
TRADING_PAIRS=BTC/USDT,ETH/USDT

# Ajustar intervalo de análisis
analysis_interval = 120  # 2 minutos en vez de 1
```

### Monitoreo

```python
from utils.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()

# Registrar eventos
tracker.record_signal('BTC/USDT', 'BUY')
tracker.record_analysis_time(1.5)  # segundos

# Obtener estadísticas
stats = tracker.get_statistics()
health = tracker.get_health_status()
```

---

Para más detalles, consulta el código fuente.
