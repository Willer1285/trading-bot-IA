# 📖 Guía de Instalación Detallada

## Índice
1. [Requisitos](#requisitos)
2. [Instalación en VPS Hostinger](#instalación-en-vps-hostinger)
3. [Configuración de Binance](#configuración-de-binance)
4. [Configuración de Telegram](#configuración-de-telegram)
5. [Configuración del Bot](#configuración-del-bot)
6. [Verificación](#verificación)

## Requisitos

### VPS Recomendado
- **CPU**: 2 vCPU mínimo
- **RAM**: 4GB mínimo (8GB recomendado)
- **Storage**: 20GB mínimo
- **OS**: Ubuntu 20.04/22.04 o Debian 11
- **Proveedor**: Hostinger, DigitalOcean, AWS, etc.

### Software
- Docker 20.10+
- Docker Compose 1.29+
- Git
- Conexión SSH al VPS

## Instalación en VPS Hostinger

### Paso 1: Conectar al VPS

```bash
ssh root@tu-ip-vps
```

### Paso 2: Actualizar Sistema

```bash
apt-get update
apt-get upgrade -y
```

### Paso 3: Ejecutar Script de Instalación

```bash
# Descargar script
wget https://raw.githubusercontent.com/tu-usuario/trading-bot-IA/main/scripts/install_vps.sh

# Dar permisos
chmod +x install_vps.sh

# Ejecutar
./install_vps.sh
```

Este script instalará:
- ✅ Docker
- ✅ Docker Compose
- ✅ Git
- ✅ Utilidades (htop, vim, etc.)

### Paso 4: Cerrar Sesión y Volver a Conectar

```bash
exit
ssh root@tu-ip-vps
```

Esto es necesario para que los permisos de Docker tomen efecto.

### Paso 5: Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/trading-bot-IA.git
cd trading-bot-IA
```

## Configuración de Binance

### Crear API Keys

1. **Iniciar sesión** en Binance.com
2. Ir a **Account > API Management**
3. Click en **Create API**
4. Completar verificación 2FA
5. Dar un nombre a la API: "Trading Bot"
6. Click en **Create**

### Configurar Permisos

✅ **Enable Reading** - Activado
✅ **Enable Spot & Margin Trading** - Activado
❌ **Enable Withdrawals** - Desactivado (por seguridad)

### Restricción de IP (Recomendado)

1. Click en **Edit restrictions**
2. Seleccionar **Restrict access to trusted IPs only**
3. Añadir la IP de tu VPS
4. Guardar

### Guardar Credenciales

⚠️ **IMPORTANTE**: Guarda tu API Key y Secret en un lugar seguro. El Secret solo se muestra una vez.

```
API Key: tu_api_key_aqui
Secret: tu_secret_aqui
```

## Configuración de Telegram

### Paso 1: Crear Bot

1. Abrir Telegram
2. Buscar `@BotFather`
3. Enviar `/newbot`
4. Seguir instrucciones:
   - Nombre del bot: "AI Trading Signals Bot"
   - Username: "ai_trading_signals_bot" (debe terminar en _bot)
5. **Guardar el token** que te da BotFather

```
Token: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
```

### Paso 2: Crear Canal

1. En Telegram, crear un nuevo canal
2. Nombre: "AI Trading Signals"
3. Hacer el canal **Privado** o **Público** (tu elección)
4. Click en **Create**

### Paso 3: Añadir Bot al Canal

1. Ir a configuración del canal
2. Click en **Administrators**
3. Click en **Add Administrator**
4. Buscar tu bot (@ai_trading_signals_bot)
5. Dar permisos de **Post messages**

### Paso 4: Obtener Channel ID

**Método 1: Usando el bot**

1. Enviar un mensaje al canal
2. Reenviar ese mensaje a `@userinfobot`
3. Te mostrará el Channel ID

**Método 2: Usando la API**

```bash
# Reemplazar TOKEN con tu token
curl https://api.telegram.org/botTOKEN/getUpdates

# Buscar "chat":{"id":-XXXXXXXXX
# El número negativo es tu Channel ID
```

Channel ID ejemplo: `-1001234567890`

## Configuración del Bot

### Paso 1: Crear Archivo .env

```bash
cd trading-bot-IA
cp .env.example .env
nano .env  # o usar vim
```

### Paso 2: Configurar Variables

```bash
# Telegram Configuration
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHANNEL_ID=-1001234567890

# Exchange API Keys (Binance)
BINANCE_API_KEY=tu_binance_api_key
BINANCE_API_SECRET=tu_binance_api_secret

# AI Model Configuration
AI_MODEL_TYPE=ensemble
CONFIDENCE_THRESHOLD=0.75
MIN_SIGNAL_SCORE=80

# Trading Pairs
TRADING_PAIRS=BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT

# Timeframes
TIMEFRAMES=1m,5m,15m,1h,4h,1d

# Risk Management
MAX_SIGNALS_PER_DAY=10
RISK_REWARD_RATIO=2.0

# Environment
TIMEZONE=UTC
ENVIRONMENT=production
```

Guardar: `Ctrl+X`, luego `Y`, luego `Enter`

### Paso 3: Configurar config.yaml (Opcional)

Para configuración avanzada:

```bash
nano config.yaml
```

Ajustar según necesidades:
- Indicadores técnicos
- Pesos del modelo ensemble
- Parámetros de riesgo
- Filtros de señales

## Verificación

### Paso 1: Verificar Configuración

```bash
# Verificar que .env existe
ls -la .env

# Verificar que tiene contenido
cat .env | grep -v "^#" | grep "="
```

### Paso 2: Iniciar Bot en Modo Test

```bash
# Dar permisos a scripts
chmod +x scripts/*.sh

# Iniciar bot
./scripts/start.sh
```

### Paso 3: Verificar Logs

El bot debería:
1. ✅ Conectarse a Binance
2. ✅ Enviar mensaje de prueba a Telegram
3. ✅ Comenzar a recopilar datos

```bash
# Ver logs en tiempo real
docker-compose logs -f trading-bot

# Buscar errores
docker-compose logs trading-bot | grep -i error

# Verificar conexión a Telegram
docker-compose logs trading-bot | grep -i telegram
```

### Paso 4: Verificar Telegram

Deberías recibir un mensaje en tu canal:

```
🚀 AI Trading Bot Started

Monitoring 5 symbols across 6 timeframes.

Environment: production
Min Confidence: 75%
Min Signal Strength: 80/100
```

## Solución de Problemas Comunes

### Error: Docker permission denied

```bash
# Añadir usuario a grupo docker
sudo usermod -aG docker $USER

# Cerrar sesión y volver a conectar
exit
ssh root@tu-ip-vps
```

### Error: Telegram bot not authorized

- Verificar que el token es correcto
- Verificar que el bot es administrador del canal
- Verificar que el Channel ID es correcto (con el signo -)

### Error: Binance API error 401

- Verificar API Key y Secret
- Verificar que la API tiene permisos habilitados
- Verificar restricciones de IP

### Bot no genera señales

Esto es normal al inicio:
- Necesita recopilar datos históricos (15-30 minutos)
- Luego comenzará a analizar
- Las señales solo se generan cuando cumplen todos los criterios

### Alto uso de memoria

```bash
# Reducir pares a monitorear en .env
TRADING_PAIRS=BTC/USDT,ETH/USDT

# O reducir timeframes
TIMEFRAMES=15m,1h,4h,1d
```

## Próximos Pasos

1. ✅ **Monitorear** el bot durante 24 horas
2. ✅ **Ajustar** parámetros según resultados
3. ✅ **Configurar backups** automáticos
4. ✅ **Revisar** señales generadas

## Mantenimiento

### Backups Diarios

```bash
# Configurar cron para backups automáticos
crontab -e

# Añadir línea (backup diario a las 00:00)
0 0 * * * cd /root/trading-bot-IA && ./scripts/backup.sh
```

### Actualizar Bot

```bash
cd trading-bot-IA
./scripts/update.sh
```

### Ver Estado

```bash
docker-compose ps
```

### Reiniciar

```bash
docker-compose restart trading-bot
```

---

¿Necesitas ayuda? Abre un issue en GitHub.
