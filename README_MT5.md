## 🤖 AI MT5 Trading Bot

Sistema avanzado de trading con **ejecución automática** en **MetaTrader 5** impulsado por Inteligencia Artificial.

Compatible con **Weltrade** y cualquier broker MT5.

---

## 🌟 Características Principales

### ✅ Ejecución Automática en MT5
- **Conexión directa** a MetaTrader 5
- **Ejecución automática** de operaciones
- **Gestión automática** de Stop Loss y Take Profit
- **Cálculo inteligente** de lotes basado en riesgo
- **Monitoreo continuo** de posiciones abiertas

### 🧠 Análisis con Inteligencia Artificial
- **+50 indicadores técnicos**
- **Modelos ensemble** (Random Forest + XGBoost + Pattern Recognition)
- **Análisis multi-timeframe** (M1, M5, M15, H1, H4, D)
- **Detección de patrones** de velas
- **Consenso multi-timeframe** para señales de alta probabilidad

### 💰 Gestión de Riesgo Profesional
- **Risk Management automático** (1% de riesgo por operación configurable)
- **Stop Loss dinámico** basado en ATR
- **Take Profit multinivel** (TP1, TP2, TP3)
- **Límite de posiciones** concurrentes
- **Filtros de calidad** de señales

### 📱 Notificaciones Telegram
- ✅ Señales de trading con gráficos
- ✅ Resúmenes diarios
- ❌ Sin confirmaciones de ejecución (solo logs internos)
- ❌ Sin actualizaciones horarias (solo logs internos)
- ℹ️ Solo se envían las señales al canal

---

## 🎯 Instrumentos Soportados

### Forex
- Pares mayores: EURUSD, GBPUSD, USDJPY, etc.
- Pares menores: EURJPY, GBPJPY, EURGBP, etc.
- Pares exóticos disponibles según broker

### Commodities
- XAUUSD (Oro)
- XAGUSD (Plata)
- USOIL (Petróleo)

### Índices
- US30 (Dow Jones)
- NAS100 (Nasdaq)
- SPX500 (S&P 500)

### Criptomonedas (si disponible en broker)
- BTCUSD
- ETHUSD

---

## 🚀 Instalación y Configuración

### Requisitos Previos

1. **MetaTrader 5** instalado y funcionando
2. **Cuenta en Weltrade** (o cualquier broker MT5)
3. **Python 3.11+** instalado
4. **Bot de Telegram** configurado

### Paso 1: Instalar Dependencias

```bash
# Clonar repositorio
git clone https://github.com/Willer1285/trading-bot-IA.git
cd trading-bot-IA

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 2: Configurar MT5

1. **Abrir MetaTrader 5**
2. **Iniciar sesión** con tu cuenta Weltrade
3. Ir a **Herramientas > Opciones > Expert Advisors**
4. Habilitar:
   - ✅ Permitir trading automático
   - ✅ Permitir importación de DLL
   - ✅ Permitir trading para Expert Advisors

### Paso 3: Configurar Variables de Entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tus credenciales
```

**Contenido del .env:**

```bash
# Telegram
TELEGRAM_BOT_TOKEN=tu_token_de_bot
TELEGRAM_CHANNEL_ID=tu_channel_id

# MT5 - Weltrade
MT5_LOGIN=12345678  # Tu número de cuenta
MT5_PASSWORD=tu_contraseña
MT5_SERVER=Weltrade-Demo  # o Weltrade-Live
MT5_PATH=  # Dejar vacío para auto-detectar

# Trading Configuration
MT5_AUTO_TRADING=true
MT5_RISK_PER_TRADE=1.0  # 1% de riesgo por operación
MT5_MAX_OPEN_POSITIONS=3

# Símbolos a monitorear
TRADING_SYMBOLS=EURUSD,GBPUSD,XAUUSD,US30

# Timeframes
TIMEFRAMES=15m,1h,4h,1d

# AI Configuration
CONFIDENCE_THRESHOLD=0.75
MIN_SIGNAL_SCORE=80
```

### Paso 4: Ejecutar el Bot

```bash
# Asegurarse que MT5 esté abierto y con sesión iniciada
python run_mt5.py
```

---

## ⚙️ Configuración Avanzada

### config.yaml

Puedes ajustar parámetros avanzados en `config.yaml`:

```yaml
mt5:
  auto_trading_enabled: true
  risk_per_trade_percent: 1.0
  max_open_positions: 3
  magic_number: 234000  # Identificador único

signals:
  confidence_threshold: 0.75  # 75% mínimo
  min_signal_score: 80

  risk_management:
    max_signals_per_day: 10
    max_signals_per_pair: 3
    min_risk_reward: 2.0
```

---

## 📊 Funcionamiento

### Flujo Automático

1. **Recopilación de Datos**
   - El bot se conecta a MT5
   - Obtiene datos históricos y en tiempo real
   - Analiza múltiples timeframes simultáneamente

2. **Análisis con IA**
   - Calcula +50 indicadores técnicos
   - Detecta patrones de velas
   - Genera predicciones con modelos ensemble
   - Calcula consenso multi-timeframe

3. **Generación de Señal**
   - Si confianza > 75% y fuerza > 80
   - Si pasa todos los filtros
   - Si no hay posiciones abiertas en ese símbolo
   - **Genera señal de trading**

4. **Ejecución Automática** ⚡
   - Calcula tamaño de lote (basado en 1% riesgo)
   - Calcula Stop Loss (ATR-based)
   - Calcula Take Profit (ratio 2:1 mínimo)
   - **Ejecuta orden en MT5**
   - Envía confirmación a Telegram

5. **Monitoreo**
   - Monitorea posiciones abiertas
   - Envía actualizaciones periódicas
   - Alerta de errores en tiempo real

---

## 📱 Ejemplo de Señal y Ejecución

### 1. Señal Generada

```
🟢 BUY SIGNAL 🟢

Symbol: EURUSD
Timeframe: 4h

📈 ENTRY
💵 Price: 1.09500

🛑 STOP LOSS
💵 Price: 1.09200

🎯 TAKE PROFIT LEVELS
   TP1: 1.10100 (+0.55%)
   TP2: 1.10400 (+0.82%)
   TP3: 1.10900 (+1.28%)

Confidence: 85%
Signal Strength: 87/100
Risk/Reward: 1:2.5

💡 Reason: 5/6 timeframes aligned, RSI oversold, MACD bullish
```

### 2. Ejecución Automática

```
✅ ORDER EXECUTED

Ticket: 123456789
Symbol: EURUSD
Type: BUY
Volume: 0.10 lots
Entry: 1.09500
SL: 1.09200
TP: 1.10100
Time: 2025-01-26 14:30:15
```

---

## 🛡️ Gestión de Riesgo

### Parámetros Configurables

- **Risk per Trade:** 1% del balance (configurable)
- **Max Positions:** 3 posiciones simultáneas
- **Stop Loss:** Automático basado en ATR
- **Take Profit:** Múltiples niveles (TP1, TP2, TP3)
- **Risk/Reward:** Mínimo 2:1

### Cálculo Automático de Lotes

El bot calcula automáticamente el tamaño del lote basado en:
- Balance de la cuenta
- Porcentaje de riesgo configurado
- Distancia del Stop Loss
- Tamaño del contrato del símbolo

**Ejemplo:**
- Balance: $10,000
- Riesgo: 1% ($100)
- Stop Loss: 30 pips
- Cálculo: ~0.33 lotes

---

## 🔧 Comandos Útiles

### Iniciar Bot

```bash
python run_mt5.py
```

### Ver Logs en Tiempo Real

```bash
tail -f logs/trading_bot.log
```

### Modo Prueba (sin ejecución automática)

Editar `.env`:
```bash
MT5_AUTO_TRADING=false
```

El bot generará señales y las enviará a Telegram, pero NO ejecutará operaciones.

---

## ⚠️ Importante

### Antes de Usar en Cuenta Real

1. ✅ **Probar en cuenta DEMO** primero
2. ✅ **Verificar todas las configuraciones**
3. ✅ **Monitorear durante 1 semana** en demo
4. ✅ **Ajustar parámetros** según resultados
5. ✅ **Empezar con riesgo bajo** (0.5-1%)

### Seguridad

- 🔐 **Nunca compartir** credenciales de MT5
- 🔒 **Usar cuenta demo** para pruebas
- 📊 **Monitorear continuamente** las primeras semanas
- 💰 **No arriesgar** más del 1-2% por operación

### Limitaciones

- Requiere que **MT5 esté abierto** y con sesión iniciada
- Funciona en **Windows, Linux, Mac** (donde MT5 esté disponible)
- Requiere **conexión a internet** estable
- **No es** una garantía de ganancias - el trading conlleva riesgos

---

## 📚 Recursos Adicionales

### Obtener Credenciales de Weltrade

1. Ir a [Weltrade.com](https://weltrade.com)
2. Abrir cuenta demo o real
3. Descargar MetaTrader 5
4. Iniciar sesión con credenciales

### Configurar Bot de Telegram

1. Buscar **@BotFather** en Telegram
2. Enviar `/newbot`
3. Seguir instrucciones
4. Copiar token
5. Crear canal y añadir bot como admin

### Obtener Channel ID

```bash
# Reemplazar TOKEN con tu token
curl https://api.telegram.org/botTOKEN/getUpdates
```

---

## 🆘 Solución de Problemas

### "MT5 initialization failed"

- ✅ Verificar que MT5 esté abierto
- ✅ Verificar credenciales en .env
- ✅ Verificar que la cuenta esté activa
- ✅ Reiniciar MT5

### "No data available"

- ✅ Esperar 30-60 segundos para recopilación inicial
- ✅ Verificar que los símbolos existan en tu broker
- ✅ Verificar conexión a internet

### "Order execution failed"

- ✅ Verificar que trading automático esté habilitado en MT5
- ✅ Verificar balance suficiente
- ✅ Verificar que el símbolo sea tradeable
- ✅ Verificar horario de trading del símbolo

### Bot no genera señales

- Normal al inicio (necesita recopilar datos)
- Las señales solo se generan cuando cumplen TODOS los criterios
- Ajustar parámetros en config.yaml si es muy restrictivo

---

## 📊 Estadísticas de Ejemplo

### Después de 1 Semana en Demo

```
📊 Weekly Summary

Total Signals: 23
Win Rate: 65%
Average R/R: 2.3:1
Profit Factor: 1.8

Best Pair: EURUSD (8 signals, 75% win)
Total Trades: 23
Winning Trades: 15
Losing Trades: 8
```

---

## 🤝 Soporte

Para problemas o preguntas:
- ✉️ Abrir un Issue en GitHub
- 📧 Contactar al desarrollador

---

## ⚖️ Disclaimer

**IMPORTANTE:** Este bot es para fines educativos y de automatización. El trading conlleva riesgos significativos de pérdida. Usa este bot bajo tu propia responsabilidad. No nos hacemos responsables de pérdidas financieras.

**Recomendaciones:**
- ✅ Siempre probar en cuenta DEMO primero
- ✅ Empezar con riesgo bajo
- ✅ Monitorear constantemente
- ✅ No invertir dinero que no puedas perder

---

## 📄 Licencia

MIT License

---

**Hecho con ❤️ para traders automatizados**

_Powered by AI & Python_ 🐍
