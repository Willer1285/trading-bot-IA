# 📝 Registro de Cambios

## v2.1.0 - Modo Canal de Señales

### ✅ Cambios Implementados

**Telegram - Solo Señales**
- ✅ El canal de Telegram **solo recibe señales** de trading
- ✅ **NO se envían confirmaciones** de ejecución de órdenes
- ✅ **NO se envían actualizaciones** horarias de balance/equity
- ✅ Canal limpio y profesional solo con señales

**Ejecución Interna**
- ✅ El bot **sigue ejecutando automáticamente** en MT5
- ✅ Todas las confirmaciones van a **logs internos**
- ✅ Detalles de tickets, volumen, precios en logs
- ✅ Sistema de monitoreo interno completo

### 📱 Qué Se Envía a Telegram

**✅ SE ENVÍA:**
```
🟢 BUY SIGNAL

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

💡 Reason: 5/6 timeframes aligned
```

**❌ NO SE ENVÍA:**
- Confirmaciones de ejecución
- Números de ticket
- Volumen ejecutado
- Actualizaciones de balance
- Actualizaciones de equity
- Estados de posiciones

### 📊 Qué Queda en Logs Internos

**Archivo: logs/trading_bot.log**

```
2025-01-26 14:30:15 | INFO | ✅ ORDER EXECUTED
  Ticket: 123456789
  Symbol: EURUSD
  Type: BUY
  Volume: 0.10 lots
  Entry: 1.09500
  SL: 1.09200
  TP: 1.10100
  Time: 2025-01-26 14:30:15

2025-01-26 15:00:00 | INFO | Hourly status
  Balance: 10150.50
  Equity: 10175.30
  Profit: 25.50
  Open Positions: 2
  Signals: 5
```

### 🎯 Beneficios

1. **Canal Profesional**
   - Solo señales limpias
   - Fácil de seguir
   - Sin ruido de confirmaciones

2. **Privacidad**
   - No se muestra cuánto se ejecuta
   - No se muestran tickets reales
   - Ideal para canales públicos/compartidos

3. **Seguimiento Interno**
   - Todo registrado en logs
   - Auditoría completa
   - Monitoreo detallado

### 🔧 Configuración

El comportamiento es automático. Si quieres cambiar:

**Habilitar confirmaciones en Telegram:**

Editar `src/main_mt5.py`, línea ~280:

```python
# Descomentar estas líneas para enviar confirmaciones
await self.telegram_bot.send_message(
    f"✅ ORDER EXECUTED\n"
    f"Ticket: {result['ticket']}\n"
    ...
)
```

**Habilitar actualizaciones horarias:**

Editar `src/main_mt5.py`, línea ~330:

```python
# Descomentar para actualizaciones horarias
await self.telegram_bot.send_message(
    f"📊 Hourly Status Update\n"
    ...
)
```

### 📁 Ver Logs

**Windows:**
```bash
type logs\trading_bot.log
```

**Linux/Mac:**
```bash
tail -f logs/trading_bot.log
```

**Ver solo ejecuciones:**
```bash
# Windows
findstr "ORDER EXECUTED" logs\trading_bot.log

# Linux/Mac
grep "ORDER EXECUTED" logs/trading_bot.log
```

---

## Versiones Anteriores

### v2.0.0 - Sistema MT5 Completo
- ✅ Integración con MetaTrader 5
- ✅ Ejecución automática de órdenes
- ✅ Compatible con Weltrade
- ✅ Gestión de riesgo automática

### v1.0.0 - Sistema Crypto Original
- ✅ Integración con exchanges crypto
- ✅ Análisis con IA
- ✅ Señales a Telegram
- ✅ Solo señales (sin ejecución)

---

**Fecha:** 26 de Octubre, 2025
**Versión Actual:** 2.1.0
