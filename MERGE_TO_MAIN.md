# ⚠️ FUSIONAR A MAIN

Todo el código del proyecto está en el branch:
```
claude/ai-trading-system-011CUVidbtpAcFVHqnwhWMeG
```

## 🔀 Cómo Fusionar a Main

### Opción 1: Desde GitHub (Recomendado)

1. Ve a: https://github.com/Willer1285/trading-bot-IA
2. Click en **"Pull requests"**
3. Click en **"New pull request"**
4. Selecciona:
   - **Base:** `main`
   - **Compare:** `claude/ai-trading-system-011CUVidbtpAcFVHqnwhWMeG`
5. Click **"Create pull request"**
6. Revisa los cambios
7. Click **"Merge pull request"**
8. Click **"Confirm merge"**

### Opción 2: Desde la Línea de Comandos

```bash
# Clonar repositorio
git clone https://github.com/Willer1285/trading-bot-IA.git
cd trading-bot-IA

# Cambiar a main
git checkout main

# Fusionar el branch de desarrollo
git merge claude/ai-trading-system-011CUVidbtpAcFVHqnwhWMeG

# Resolver conflictos si hay (usar nuestro README)
git checkout --ours README.md
git add README.md
git commit -m "Merge development branch"

# Push a main
git push origin main
```

### Opción 3: Reemplazar Main Completamente

Si quieres reemplazar main completamente con el contenido del branch:

```bash
git checkout main
git reset --hard claude/ai-trading-system-011CUVidbtpAcFVHqnwhWMeG
git push origin main --force
```

⚠️ **CUIDADO:** Esto sobrescribirá todo en main.

---

## 📁 Contenido del Proyecto

Una vez fusionado a main, tendrás:

```
trading-bot-IA/
├── src/                     # Código fuente
│   ├── main_mt5.py         # App principal MT5
│   ├── ai_engine/          # Motor de IA
│   ├── data_collector/     # Datos MT5
│   ├── signal_generator/   # Señales
│   └── telegram_bot/       # Telegram
├── requirements.txt         # Dependencias
├── config.yaml             # Configuración
├── run_mt5.py              # Launcher
├── README_MT5.md           # Documentación
├── QUICK_START_MT5.md      # Guía rápida
└── CHANGELOG.md            # Cambios
```

## 🚀 Siguiente Paso

Después de fusionar a main, puedes:

1. **Clonar desde main:**
   ```bash
   git clone https://github.com/Willer1285/trading-bot-IA.git
   ```

2. **Instalar y usar:**
   ```bash
   cd trading-bot-IA
   pip install -r requirements.txt
   cp .env.example .env
   # Editar .env con tus credenciales
   python run_mt5.py
   ```

---

**Branch actual con todo el código:**
`claude/ai-trading-system-011CUVidbtpAcFVHqnwhWMeG` ✅
