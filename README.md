# 🤖 AI Trading Signal Bot

Sistema avanzado de trading impulsado por IA que analiza mercados en tiempo real y envía señales de alta probabilidad a Telegram.

## 🌟 Características

### Análisis de Mercado
- ✅ **Recopilación de datos en tiempo real** de múltiples exchanges (Binance, Bybit)
- ✅ **Análisis multi-timeframe** (M1, M5, M15, H1, H4, D)
- ✅ **+50 indicadores técnicos** (RSI, MACD, Bollinger Bands, ATR, EMA, etc.)
- ✅ **Detección de patrones** de velas japonesas
- ✅ **Análisis de soporte y resistencia**

### Inteligencia Artificial
- 🧠 **Modelo ensemble** combinando Random Forest, Gradient Boosting y análisis de patrones
- 📊 **Ingeniería de características** avanzada con +100 features
- 🎯 **Consenso multi-timeframe** para señales de alta probabilidad
- 🔍 **Análisis de régimen de mercado** (trending vs ranging)

### Gestión de Riesgo
- 💰 **Stop Loss automático** basado en ATR y niveles de soporte/resistencia
- 🎯 **Múltiples niveles de Take Profit** (TP1, TP2, TP3)
- 📈 **Ratio Risk/Reward** mínimo configurable
- 🔒 **Filtros de señales** para evitar operaciones de baja calidad
- ⚖️ **Límites diarios** de señales por par y total

### Telegram
- 📱 **Señales formateadas** con toda la información necesaria
- 📊 **Gráficos automáticos** con indicadores y niveles
- 📈 **Resúmenes diarios** de rendimiento
- ⚠️ **Alertas de errores** en tiempo real
- 💬 **Mensajes personalizables**

### Operación 24/7
- 🐳 **Docker & Docker Compose** para deployment fácil
- 🔄 **Auto-restart** en caso de errores
- 📝 **Logging completo** con rotación automática
- 📊 **Monitoreo de rendimiento** y health checks
- 💾 **Backups automáticos** de datos y logs

## 🚀 Instalación Rápida

### Requisitos Previos
- VPS con Ubuntu/Debian (Hostinger u otro)
- Docker y Docker Compose
- Cuenta en Binance (API keys)
- Bot de Telegram (BotFather)

### Instalación en VPS

```bash
# 1. Clonar el repositorio
git clone <your-repo-url>
cd trading-bot-IA

# 2. Ejecutar script de instalación
chmod +x scripts/install_vps.sh
./scripts/install_vps.sh

# 3. Configurar variables de entorno
cp .env.example .env
nano .env  # Editar con tus credenciales

# 4. Iniciar el bot
./scripts/start.sh
```

## ⚙️ Configuración

### 1. Variables de Entorno (.env)

```bash
# Telegram
TELEGRAM_BOT_TOKEN=tu_token_de_bot
TELEGRAM_CHANNEL_ID=tu_channel_id

# Binance API
BINANCE_API_KEY=tu_api_key
BINANCE_API_SECRET=tu_api_secret

# Configuración de IA
CONFIDENCE_THRESHOLD=0.75  # 75% mínimo
MIN_SIGNAL_SCORE=80  # 0-100

# Pares a monitorear
TRADING_PAIRS=BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT

# Timeframes
TIMEFRAMES=1m,5m,15m,1h,4h,1d
```

### 2. Configuración Avanzada (config.yaml)

El archivo `config.yaml` permite configuración detallada:

```yaml
ai_model:
  confidence_threshold: 0.75
  min_signal_score: 80

signals:
  risk_management:
    max_signals_per_day: 10
    max_signals_per_pair: 3
    min_risk_reward: 2.0
    stop_loss_atr_multiplier: 1.5
```

## 🎯 Uso

### Iniciar el Bot

```bash
./scripts/start.sh
```

### Ver Logs

```bash
docker-compose logs -f trading-bot
```

### Detener el Bot

```bash
./scripts/stop.sh
```

### Actualizar

```bash
./scripts/update.sh
```

### Crear Backup

```bash
./scripts/backup.sh
```

## 📊 Señales de Trading

Las señales incluyen:

- 🎯 **Tipo de señal**: BUY o SELL
- 💵 **Precio de entrada**
- 🛑 **Stop Loss** (automático basado en ATR)
- 🎯 **Take Profit** (3 niveles: TP1, TP2, TP3)
- 📊 **Confianza** (0-100%)
- 💪 **Fuerza de señal** (0-100)
- 📈 **Ratio Risk/Reward**
- 💡 **Razón** de la señal (indicadores alineados)
- 📊 **Gráfico** con niveles marcados

### Ejemplo de Señal

```
🟢 BUY SIGNAL 🟢

Symbol: BTC/USDT
Timeframe: 4h

📈 ENTRY
💵 Price: $42,500.00

🛑 STOP LOSS
💵 Price: $41,800.00

🎯 TAKE PROFIT LEVELS
   TP1: $43,900.00 (+3.29%)
   TP2: $44,600.00 (+4.94%)
   TP3: $45,800.00 (+7.76%)

Confidence: 85%
Signal Strength: 87/100
Risk/Reward: 1:2.5

💡 Reason: 5/6 timeframes aligned, RSI oversold, MACD bullish
```

## 🏗️ Arquitectura

```
trading-bot-IA/
├── src/
│   ├── main.py                    # Aplicación principal
│   ├── config.py                  # Gestión de configuración
│   ├── data_collector/            # Recopilación de datos
│   │   ├── exchange_connector.py
│   │   ├── market_data_manager.py
│   │   └── timeframe_aggregator.py
│   ├── ai_engine/                 # Motor de IA
│   │   ├── feature_engineering.py
│   │   ├── technical_indicators.py
│   │   ├── ai_models.py
│   │   └── market_analyzer.py
│   ├── signal_generator/          # Generación de señales
│   │   ├── signal_generator.py
│   │   ├── signal_filter.py
│   │   └── risk_manager.py
│   ├── telegram_bot/              # Bot de Telegram
│   │   ├── telegram_bot.py
│   │   ├── message_formatter.py
│   │   └── chart_generator.py
│   └── utils/                     # Utilidades
│       ├── logger.py
│       └── performance_tracker.py
├── scripts/                       # Scripts de deployment
├── models/                        # Modelos entrenados
├── logs/                          # Logs
├── data/                          # Datos temporales
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── config.yaml
```

## 🔧 Características Técnicas

### Indicadores Técnicos (50+)
- **Tendencia**: SMA, EMA, MACD, ADX, Ichimoku
- **Momentum**: RSI, Stochastic, Williams %R, ROC, TSI
- **Volatilidad**: Bollinger Bands, ATR, Keltner Channel
- **Volumen**: OBV, CMF, VWAP, Force Index

### Modelos de IA
- **Random Forest**: Clasificación basada en árboles de decisión
- **Gradient Boosting**: Boosting para patrones complejos
- **Pattern Recognition**: Análisis de patrones de velas
- **Ensemble**: Combinación ponderada de todos los modelos

### Gestión de Riesgo
- Stop Loss basado en ATR con multiplicador configurable
- Consideración de niveles de soporte/resistencia
- Múltiples niveles de Take Profit
- Filtrado de señales de baja calidad
- Límites de exposición diaria

## 📈 Rendimiento

El sistema está diseñado para generar señales de **alta probabilidad** con:
- ✅ Confianza mínima: 75%
- ✅ Fuerza de señal mínima: 80/100
- ✅ Risk/Reward mínimo: 2:1
- ✅ Máximo 10 señales por día
- ✅ Máximo 3 señales por par

## 🔒 Seguridad

- 🔐 Variables de entorno para credenciales
- 🚫 No almacena claves privadas
- 📝 Logging completo de operaciones
- 🔄 Auto-restart en caso de errores
- 💾 Backups automáticos

## 🛠️ Solución de Problemas

### El bot no inicia
```bash
# Verificar logs
docker-compose logs trading-bot

# Verificar configuración
cat .env

# Reiniciar servicios
docker-compose restart
```

### No se envían señales a Telegram
```bash
# Verificar token de Telegram
# Verificar que el bot sea administrador del canal
# Verificar logs para errores
docker-compose logs trading-bot | grep -i telegram
```

### Errores de API del exchange
```bash
# Verificar API keys en .env
# Verificar límites de rate
# Verificar conectividad
docker-compose logs trading-bot | grep -i error
```

## 📚 Recursos

### Crear Bot de Telegram
1. Buscar @BotFather en Telegram
2. Enviar `/newbot`
3. Seguir instrucciones
4. Copiar el token

### Obtener Channel ID
1. Crear un canal
2. Añadir el bot como administrador
3. Enviar mensaje al canal
4. Visitar: `https://api.telegram.org/bot<TOKEN>/getUpdates`
5. Buscar `"chat":{"id":-XXXXXXXXX`

### API de Binance
1. Ir a Binance.com
2. Account > API Management
3. Crear nueva API Key
4. Habilitar "Enable Spot & Margin Trading"
5. Guardar API Key y Secret

## 🤝 Contribuir

Las contribuciones son bienvenidas! Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## ⚠️ Disclaimer

Este bot es para fines educativos. El trading conlleva riesgos significativos. Usa este bot bajo tu propia responsabilidad. No nos hacemos responsables de pérdidas financieras.

**IMPORTANTE**: Siempre haz backtesting y paper trading antes de usar con dinero real.

## 📄 Licencia

MIT License - Ver LICENSE file para detalles

## 💬 Soporte

Para problemas o preguntas:
- Abre un Issue en GitHub
- Contacta al desarrollador

## 🙏 Agradecimientos

- CCXT por la conectividad con exchanges
- TA-Lib por indicadores técnicos
- python-telegram-bot por la integración con Telegram
- Scikit-learn por los modelos de ML

---

**Hecho con ❤️ para la comunidad de trading**

_Powered by AI & Python_ 🐍
