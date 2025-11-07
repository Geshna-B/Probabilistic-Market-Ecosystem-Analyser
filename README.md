# Probabilistic-Market-Ecosystem-Analyser

**An Integrated Framework for Financial Market Analysis Using Regime Detection and Ecological Modeling**

## Overview

**Probabilistic-Market-Ecosystem-Analyser** is an advanced financial analytics platform that integrates **Markov regime detection**, **Monte Carlo simulations**, and **Lotka-Volterra ecological modeling** to provide a unified probabilistic framework for market analysis.
It combines rigorous mathematical models with interactive web-based visualizations to capture **nonlinear market dynamics**, **regime changes**, and **buyer–seller interactions** in modern financial ecosystems.



## Key Features

* **Regime Detection Engine:**
  Detects Bull, Bear, and Stable market states using rolling z-score analysis and Markov modeling.

* **Monte Carlo Simulation:**
  Enhanced Geometric Brownian Motion with volatility-adjusted parameter clipping and realistic boundary conditions for 60-day probabilistic forecasts.

* **Ecological Market Modeling:**
  Implements Lotka–Volterra equations to represent buyer–seller dynamics (prey–predator model), providing insights into market stability and transitions.

* **Risk Management Module:**
  Includes **Value at Risk (VaR)** and **Conditional VaR (CVaR)** calculations for quantitative risk assessment.

* **AI-Generated Insights:**
  Translates complex quantitative results into natural language summaries and actionable investment insights.

* **Web Interface:**
  Built using **Streamlit** and **Plotly**, enabling interactive analysis, real-time parameter tuning, and intuitive visual dashboards for non-technical users.



## System Architecture

The system integrates three primary analytical engines within a unified workflow:

1. **Regime Detection Engine** – Identifies market states based on rolling statistical metrics.
2. **Monte Carlo Simulation Engine** – Generates stochastic price paths and risk metrics.
3. **Ecological Modeling Engine** – Analyzes market pressure dynamics using differential equations.

Each module feeds into a **central dashboard** for synchronized visualization and comparative analytics.


## Mathematical Foundations

| **Model**                           | **Purpose**                                                                  | **Core Equation / Description**                                                    |
| :---------------------------------- | :--------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- |
| **Markov Regime Detection**         | Classifies market states (Bull, Bear, Stable) using rolling z-score analysis | Transition probabilities estimated from time-series data                           |
| **Monte Carlo Simulation (GBM)**    | Simulates probabilistic future price paths                                   | ( dS = \mu S , dt + \sigma S , dW )                                                |
| **Lotka–Volterra Ecological Model** | Models interaction between buying (prey) and selling (predator) pressures    | ( \frac{dx}{dt} = \alpha x - \beta xy ),  ( \frac{dy}{dt} = \delta xy - \gamma y ) |
| **Risk Metrics (VaR & CVaR)**       | Quantifies potential losses under uncertainty                                | Value at Risk (VaR), Conditional VaR for tail-risk estimation                      |


## Experimental Results

* Achieved **high accuracy** in regime classification on historical datasets.
* **Low forecast error** on 60-day projections using Monte Carlo simulations.
* Ecological model successfully detected **early warning signals** for market turning points.
* Demonstrated **robust parameter stability** and **interpretable outputs**.



## Tech Stack

| **Component**                           | **Technology Used**                           |
| :-------------------------------------- | :-------------------------------------------- |
| **Frontend**                            | Streamlit, Plotly                             |
| **Backend / Core Logic**                | Python (NumPy, Pandas, SciPy, Statsmodels)    |
| **Data Handling**                       | Yahoo Finance API, CSV Historical Data        |
| **Mathematical & Statistical Modeling** | Scikit-learn, DifferentialEquations library   |
| **Visualization & Analytics**           | Matplotlib, Seaborn, Plotly                   |
| **AI & NLP Layer (Optional)**           | OpenAI API, LangChain for generating insights |
| **Risk Analysis**                       | Custom VaR & CVaR computation modules         |
| **Deployment**                          | Streamlit Cloud / Localhost                   |



## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Probabilistic-Market-Ecosystem-Analyser.git

# Navigate to the project directory
cd Probabilistic-Market-Ecosystem-Analyser

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # for Mac/Linux
venv\Scripts\activate     # for Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```



## Usage

1. Upload or fetch historical financial data.
2. Select the analysis module (Regime Detection / Monte Carlo / Ecological).
3. Adjust model parameters interactively.
4. Visualize forecasts, stability regions, and risk metrics.
5. Access AI-generated plain-language insights for actionable decisions.

