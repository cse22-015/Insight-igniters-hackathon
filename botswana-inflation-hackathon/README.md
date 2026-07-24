Botswana Food Price Inflation Forecasting System

-IndabaX Botswana Hackathon 2026 Submission**  
Team:Insight Igniters  
Target Variable: FAO Item Code 23014 — Food Price Inflation (% Year-on-Year) [Dataset 4]  
Forecast Horizon: January 2024 – December 2024 (12 Months)  


Project Overview

This repository contains the complete two-model forecasting system developed for the IndabaX Botswana 2026 Hackathon. To evaluate macroeconomic shock transmission (oil price spikes, maritime freight volatility, interest rate shifts) onto domestic human capital, we constructed and benchmarked two distinct forecasting architectures:

1. Model 1 (Classical Baseline): XGBoost Regressor with Augmented Dickey-Fuller stationarity checks and dynamic lag engineering.
2. Model 2 (Deep Learning): PyTorch Long Short-Term Memory (LSTM) Recurrent Neural Network designed with regularisation for small sample time-series data ($N \approx 288$).

Both models are strictly trained on data ending December 2023 to prevent phantom 2024 feature leakage, adhering to Strategy (a): Lagged Features & Autoregressive Rolling Projection.


Installed Dependencies (`requirements.txt`)

This project relies on the following Python packages:

pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
xgboost>=1.7.0
torch>=2.0.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0

Installation & Setup Instructions
You can run this project using either uv or standard Python venv

Option 1: Fast Execution using uv
uv automatically manages environments and dependencies:

-Clone the repository
git clone [https://github.com/cse22-015/Insight-igniters-hackathon.git](https://github.com/cse22-015/Insight-igniters-hackathon.git)
cd Insight-igniters-hackathon

-Execute scripts directly using uv
uv run 1_classical_baseline.py
uv run 2_deep_learning_lstm.py
uv run 3_generate_2024_forecast.py
uv run 4_hcp_linkage_analysis.py

Option 2: Standard Python Virtual Environment (venv)

Clone the repository
git clone [https://github.com/cse22-015/Insight-igniters-hackathon.git](https://github.com/cse22-015/Insight-igniters-hackathon.git)
cd Insight-igniters-hackathon

- Create and activate virtual environment
python -m venv .venv

- Activate environment (Windows)
.venv\Scripts\activate

- Install dependencies
pip install -r requirements.txt

- Run scripts sequentially
python 1_classical_baseline.py
python 2_deep_learning_lstm.py
python 3_generate_2024_forecast.py
python 4_hcp_linkage_analysis.py

Step-by-Step Execution Guide
Step 1: Classical Baseline Model (XGBoost)

Calculates stationarity, fits XGBoost regressor, evaluates cross-validation error, and generates diagnostic plots: uv run 1_classical_baseline.py

Step 2: Deep Learning Model (PyTorch LSTM)

Formats time series sequences, trains the LSTM neural network, and computes out-of-sample loss metrics:  uv run 2_deep_learning_lstm.py

Step 3: Export 2024 Predictions (Deliverable 1.1a)

Runs the 12-month ahead autoregressive forecasting loop for 2024 using the winning XGBoost baseline model and exports best_model_predictions.csv:

uv run 3_generate_2024_forecast.py

Step 4: Human Capital Project (HCP) Statistical AnalysisExecutes statistical regressions and outputs performance metrics ($R^2$, $p$-values, coefficients) for human capital linkages:  uv run 4_hcp_linkage_analysis.py


		
