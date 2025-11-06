# Fake News Circulation Simulator

A simulation tool for studying the spread of fake news using both Agent-Based Model (ABM) and Population-Based Model (PBM) approaches.

## Features

- Dual simulation models: ABM and PBM
- Real-time visualization of news spread
- Interactive parameter adjustment
- Detailed summary and comparison analysis
- Support for different types of fake news scenarios

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

## Project Structure

- `src/` - Source code files
  - `models/` - Simulation models (ABM and PBM)
  - `gui/` - GUI components
  - `utils/` - Utility functions
- `data/` - Input data files
- `results/` - Simulation results and visualizations
- `tests/` - Unit tests
- `visualization/` - Visualization components

## License

MIT License