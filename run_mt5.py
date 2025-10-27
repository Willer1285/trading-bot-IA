"""
Launcher for MT5 Trading Bot
Quick start script for the MT5 version
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main_mt5 import main

if __name__ == "__main__":
    print("=" * 60)
    print("AI MT5 Trading Bot")
    print("Starting...")
    print("=" * 60)
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
