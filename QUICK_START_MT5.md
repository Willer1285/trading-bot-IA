# 🚀 Guía Rápida - MT5 Trading Bot

## Instalación en Windows (Recomendado)

### Paso 1: Descargar e Instalar MT5

1. Ir a [Weltrade.com](https://weltrade.com)
2. Descargar **MetaTrader 5**
3. Instalar MT5
4. Abrir cuenta **Demo** para pruebas
5. **Iniciar sesión** en MT5 y dejarlo abierto

### Paso 2: Instalar Python

1. Descargar [Python 3.11](https://www.python.org/downloads/)
2. ✅ **IMPORTANTE**: Marcar "Add Python to PATH"
3. Instalar

### Paso 3: Configurar Bot

```bash
# Abrir PowerShell o CMD

# Clonar repositorio
git clone https://github.com/Willer1285/trading-bot-IA.git
cd trading-bot-IA

# Crear entorno virtual
python -m venv venv

# Activar entorno
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 4: Configurar Credenciales

```bash
# Copiar archivo de configuración
copy .env.example .env

# Editar con Notepad
notepad .env
```

**Configurar en .env:**

```bash
# Telegram (crear bot con @BotFather)
TELEGRAM_BOT_TOKEN=tu_token_aqui
TELEGRAM_CHANNEL_ID=tu_channel_id_aqui

# MT5 - Tus credenciales de Weltrade
MT5_LOGIN=12345678
MT5_PASSWORD=tu_contraseña
MT5_SERVER=Weltrade-Demo
MT5_PATH=

# Trading automático (false para solo señales, true para ejecutar)
MT5_AUTO_TRADING=false

# Símbolos (puedes cambiar los que quieras)
TRADING_SYMBOLS=EURUSD,GBPUSD,XAUUSD

# Timeframes
TIMEFRAMES=15m,1h,4h,1d
```

### Paso 5: Habilitar Trading Automático en MT5

1. En MT5: **Herramientas > Opciones > Expert Advisors**
2. Marcar:
   - ✅ Permitir trading automático
   - ✅ Permitir importación de DLL
   - ✅ Permitir trading para Expert Advisors
3. Click en **OK**

### Paso 6: Iniciar Bot

```bash
# Asegurarse que MT5 esté abierto
python run_mt5.py
```

## ✅ Verificación

Deberías ver:

```
AI MT5 Trading Bot
Starting...

MT5 initialized successfully
Connected to MT5 - Account: 12345678, Balance: 10000.0, Server: Weltrade-Demo
Telegram bot initialized
Bot started successfully!
```

Y recibir mensaje en Telegram:

```
🚀 AI MT5 Trading Bot Started

Account: 12345678
Balance: 10000.0 USD
Server: Weltrade-Demo

Monitoring 3 symbols: EURUSD, GBPUSD, XAUUSD
...
```

## 🎯 Modos de Operación

### Modo 1: Solo Señales (Recomendado al inicio)

```bash
# En .env
MT5_AUTO_TRADING=false
```

- ✅ Genera señales
- ✅ Envía a Telegram
- ❌ NO ejecuta automáticamente

### Modo 2: Trading Automático

```bash
# En .env
MT5_AUTO_TRADING=true
MT5_RISK_PER_TRADE=1.0
```

- ✅ Genera señales
- ✅ Envía señales a Telegram
- ✅ **EJECUTA AUTOMÁTICAMENTE** (sin enviar confirmación a Telegram)

⚠️ **SOLO usar en cuenta DEMO primero!**

## 📊 Configurar Símbolos

Editar `.env` o `config.yaml`:

### Forex
```bash
TRADING_SYMBOLS=EURUSD,GBPUSD,USDJPY,AUDUSD
```

### Oro y Commodities
```bash
TRADING_SYMBOLS=XAUUSD,XAGUSD,USOIL
```

### Índices
```bash
TRADING_SYMBOLS=US30,NAS100,SPX500
```

### Mix
```bash
TRADING_SYMBOLS=EURUSD,XAUUSD,US30,BTCUSD
```

## 🔧 Ajustar Parámetros

### Riesgo por Operación

```bash
# .env
MT5_RISK_PER_TRADE=1.0  # 1% del balance
```

### Máximo de Posiciones

```bash
# .env
MT5_MAX_OPEN_POSITIONS=3  # 3 operaciones simultáneas
```

### Confianza Mínima

```bash
# .env
CONFIDENCE_THRESHOLD=0.75  # 75%
MIN_SIGNAL_SCORE=80  # 80/100
```

## 📱 Configurar Telegram

### 1. Crear Bot

1. Buscar **@BotFather** en Telegram
2. Enviar: `/newbot`
3. Nombre: `My Trading Bot`
4. Username: `my_trading_bot_123_bot`
5. **Copiar token**

### 2. Crear Canal

1. Crear nuevo canal
2. Hacerlo privado
3. Añadir tu bot como administrador
4. Dar permiso de "Post messages"

### 3. Obtener Channel ID

```bash
# En navegador (reemplazar TOKEN con tu token)
https://api.telegram.org/botTOKEN/getUpdates

# Enviar mensaje al canal primero
# Buscar "chat":{"id":-XXXXXXXXX
# Copiar el número (con el signo -)
```

## ❓ Preguntas Frecuentes

### ¿Necesito dejar MT5 abierto?

✅ **SÍ** - MT5 debe estar abierto y con sesión iniciada

### ¿Funciona en cuenta real?

✅ Sí, pero **PRUEBA EN DEMO PRIMERO**

### ¿Cuánto tiempo antes de ver señales?

⏱️ 30-60 minutos para recopilar datos iniciales

### ¿Cómo detener el bot?

`Ctrl+C` en la consola

### ¿El bot pierde dinero?

⚠️ El trading conlleva riesgos. **NO es garantía de ganancias.**

### ¿Puedo usar otro broker?

✅ Sí, cualquier broker MT5. Solo cambia el servidor en `.env`

## 🆘 Solución de Problemas

### "MT5 initialization failed"

✅ Verificar que MT5 esté abierto
✅ Verificar que hayas iniciado sesión
✅ Reiniciar MT5

### "No data available"

✅ Esperar 1-2 minutos
✅ Verificar que el símbolo exista en tu broker
✅ Verificar internet

### "Telegram error"

✅ Verificar token
✅ Verificar que el bot sea admin del canal
✅ Verificar Channel ID (con signo -)

### "Order execution failed"

✅ Verificar "Trading automático" habilitado en MT5
✅ Verificar balance suficiente
✅ Verificar horario de trading del símbolo

## 📞 Soporte

- GitHub Issues
- Telegram (si configurado)
- Email del desarrollador

## ⚖️ Disclaimer

Este bot es para **fines educativos**. El trading conlleva **riesgos significativos**.

✅ Probar en DEMO primero
✅ No invertir dinero que no puedas perder
✅ El rendimiento pasado no garantiza resultados futuros

---

**¡Listo para empezar! 🚀**
