# DexterioBOT (Refactored)

This is the refactored **DexterioBOT** trading system from 2025. The project is organized into modules as detailed below.

## Structure

- **core/**: Core libraries for data handling, strategy logic, risk management, execution engine, and reporting.
- **models/**: Data models for setup scoring (CJR logic).
- **main.py**: Entry point to run the backtest.
- **backtest.py**: Executable wrapper (calls `main()`).
- **journal/**: Output folder created at runtime containing CSV trade journals.
- **plots/**: Output folder for equity curve PNGs.
- **config.json**: Configuration file (symbols, timeframe, strategy, risk profile, etc.).
- **.env**: (Optional) File containing `DISCORD_WEBHOOK_URL` for Discord reports.

## Usage

1. Edit `config.json` with desired symbols, timeframe, lookback, strategy mode, and risk profile.
2. Ensure any required environment variables (e.g. `DISCORD_WEBHOOK_URL`) are set.
3. Run the script:
   ```bash
   python backtest.py
