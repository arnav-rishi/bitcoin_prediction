# Bitcoin Price Prediction using LSTM with Technical Indicators

Predicting the future price of Bitcoin (BTC) or any cryptocurrency is highly speculative and subject to a wide range of factors, including market sentiment, regulatory developments, technological advancements, macroeconomic trends, and more. This project implements a Long Short-Term Memory (LSTM) neural network enhanced with technical indicators to predict Bitcoin price movements.

## Important Disclaimer

- **Volatility:** Bitcoin is known for its extreme price volatility. Prices can fluctuate significantly in short periods, making accurate predictions challenging.

- **Market Sentiment:** Investor sentiment plays a crucial role. Positive news (like institutional adoption or regulatory clarity) tends to drive prices up, while negative news (like regulatory crackdowns or security breaches) can lead to sharp declines.

- **Technological Developments:** Upgrades to the Bitcoin network, such as improvements in scalability (like the Lightning Network) or changes in mining technology, can influence price movements.

- **Macroeconomic Factors:** Bitcoin is often seen as a hedge against inflation and currency devaluation. Economic events, such as changes in interest rates or geopolitical tensions, can impact its price.

- **Regulatory Environment:** Government regulations and policies regarding cryptocurrencies can have a significant impact on their adoption and, consequently, their value.

Given these complexities, making precise predictions for Bitcoin's price is challenging. This model achieved a modest R² score of 0.016, indicating limited but positive predictive power. It's important to approach any predictions with caution and consider a diverse range of viewpoints and analysis from financial experts and analysts.

## Overview

This repository contains a deep learning model for predicting Bitcoin (BTC) price movements using Long Short-Term Memory (LSTM) neural networks enhanced with technical indicators. The model analyzes historical price data, trading volume, and technical indicators to forecast daily percentage changes in Bitcoin prices.

### Key Highlights

- **Model Architecture:** 2-layer LSTM with 100 units each, using a 60-day sequence window
- **Technical Indicators:** Simple Moving Average (SMA), Relative Strength Index (RSI), Volume percentage change, and SMA difference
- **Dataset:** Daily Bitcoin price data from October 2014 to November 2025 (over 4,000 data points)
- **Performance:** Achieved R² of 0.016, outperforming baseline models (ARIMA, Linear Regression, Random Forest, XGBoost)

## Key Features

- **Data Collection & Preprocessing:** 
  - Historical data from CSV files
  - Real-time data fetching from Yahoo Finance API
  - Data merging and validation

- **Feature Engineering:**
  - 15-day Simple Moving Average (SMA)
  - 14-day Relative Strength Index (RSI)
  - Volume percentage change
  - Normalized SMA difference

- **LSTM Neural Network Model:**
  - 2 LSTM layers (100 units each)
  - Dropout regularization (0.2)
  - 60-day sequence window for temporal pattern learning
  - Early stopping to prevent overfitting

- **Model Evaluation:**
  - Comprehensive metrics: R², MAE, RMSE
  - Comparison with baseline models (ARIMA, Linear Regression, Random Forest, XGBoost)
  - 5-fold time series cross-validation
  - Hyperparameter tuning results

- **Visualization:**
  - Interactive HTML dashboard with Plotly charts
  - Time series plots, distribution histograms, correlation heatmaps
  - Model performance comparisons
  - System architecture and data flow diagrams

## Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Required Libraries

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `tensorflow` (2.12.0+) - Deep learning framework
- `pandas` (1.5.0+) - Data manipulation
- `numpy` (1.23.0+) - Numerical computing
- `scikit-learn` (1.2.0+) - Machine learning utilities
- `yfinance` (0.2.0+) - Yahoo Finance data fetching
- `matplotlib` (3.7.0+) - Plotting
- `plotly` (5.14.0+) - Interactive visualizations

### Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd bitcoin_prediction_optimized
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open and run the `bitcoin_prediction_optimized.ipynb` notebook.

## Usage

### Training the Model

1. Open the Jupyter Notebook `bitcoin_prediction_optimized.ipynb`
2. Run the cells in sequence:
   - **Cell 1:** Load and merge historical data with latest data from Yahoo Finance
   - **Cell 3:** Preprocess data and engineer features (SMA, RSI, Volume changes)
   - **Cell 5:** Create sequences with 60-day windows
   - **Cell 8:** Build and compile the LSTM model
   - **Cell 9:** Train the model with early stopping

### Making Predictions

- The trained model can predict daily percentage changes in Bitcoin prices
- Predictions are made on the test set (20% of data)
- Results are inverse-transformed to original scale for interpretation

### Evaluation

The notebook includes comprehensive evaluation:
- **Cell 12:** Calculate metrics (R², MAE, RMSE) on test data
- **Cell 16:** Compare with ARIMA baseline model
- **Cell 19:** Compare with other machine learning models
- **Cell 22:** 5-fold time series cross-validation
- **Cell 25:** Hyperparameter tuning with grid search

### Visualization

- **HTML Dashboard:** Open `visualizations.html` in a web browser to view interactive charts
- **Generate Plots:** Run `generate_all_plots.py` to create static visualizations:
  ```bash
  python generate_all_plots.py
  ```
- **Generate Diagrams:** Run `generate_diagrams.py` to create architecture diagrams:
  ```bash
  python generate_diagrams.py
  ```

## Model Architecture

```
Input Layer: (60, 4) - 60 days of historical data with 4 features
    ↓
LSTM Layer 1: 100 units, return_sequences=True
    ↓
Dropout: Rate = 0.2
    ↓
LSTM Layer 2: 100 units, return_sequences=False
    ↓
Dropout: Rate = 0.2
    ↓
Dense Output Layer: 1 unit (predicted percentage change)
```

## Features Used

1. **Close Price:** Daily closing price of Bitcoin
2. **Volume Percentage Change:** Percentage change in trading volume
3. **SMA Difference:** Normalized difference between current price and 15-day moving average
4. **RSI:** Relative Strength Index calculated over 14 days

## Results

### Model Performance

| Model | R² | MAE (% change) | RMSE (% change) |
|-------|----|----------------|-----------------|
| **LSTM (Our Model)** | **0.016** | **0.028074** | **0.038500** |
| ARIMA(1,0,1) | -0.001 | 0.028032 | 0.038841 |
| Random Forest | -0.936 | 0.043203 | 0.054007 |
| XGBoost | -1.422 | 0.046575 | 0.060401 |
| Linear Regression | -3.781 | 0.063332 | 0.084867 |

### Key Findings

- The LSTM model achieved a positive R² score (0.016), indicating it found a real predictive signal
- Outperformed all baseline models in terms of R² score
- Technical indicators (SMA, RSI, Volume) provided meaningful signals for prediction
- The 60-day sequence window effectively captured medium-term patterns

## Project Structure

```
bitcoin_prediction_optimized/
├── bitcoin_prediction_optimized.ipynb  # Main Jupyter notebook
├── btc.csv                             # Historical Bitcoin dataset
├── visualizations.html                 # Interactive dashboard
├── generate_price_plot.py              # Script to generate price plots
├── generate_all_plots.py              # Script to generate all visualizations
├── generate_diagrams.py               # Script to generate architecture diagrams
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Generated Visualizations

The project includes scripts to generate various visualizations:
- Bitcoin price history time series
- Price distribution histogram
- Feature correlation heatmap
- System architecture diagram
- Data flow diagram
- LSTM architecture diagram

## Contributing

Contributions to improve the project are welcome! Here are a few ways you can contribute:

- Implement additional technical indicators
- Experiment with different sequence lengths
- Try other deep learning architectures (GRU, Transformer)
- Enhance data preprocessing techniques
- Optimize hyperparameters and improve model accuracy
- Add more comprehensive evaluation metrics
- Implement ensemble methods

If you find any issues or have suggestions, please open an issue or create a pull request.

## Limitations

- The model explains only 1.6% of the variance (R² = 0.016), highlighting the difficulty of cryptocurrency prediction
- Cannot account for unexpected events (regulatory changes, major news, market manipulation)
- Performance may degrade during unprecedented market conditions
- Cross-validation results show limited generalizability across different time periods

## Future Improvements

- Integrate sentiment analysis from social media and news
- Include correlations with other cryptocurrencies and traditional assets
- Explore transformer architectures for time series
- Implement reinforcement learning-based trading strategies
- Develop mechanisms for continuous model updating with new data

## Contact

For questions or discussions about this project:

- **Institution:** Vellore Institute of Technology
- **Department:** School of Computer Science Engineering & Information Systems
- **Course:** B.Tech. - Information Technology
- **Subject:** Artificial Intelligence Theory (BITE308L)

## License

This project is for educational purposes. Please use responsibly and understand that cryptocurrency price prediction involves significant risk.

## Acknowledgments

- Dr. S. Hemalatha for project guidance
- School of Computer Science Engineering & Information Systems, VIT
- Open source community for excellent libraries and frameworks
