# Stock Price Prediction Using LSTM Neural Networks

## Description
This project aims to predict future stock prices using Long Short-Term Memory (LSTM) neural networks, a type of Recurrent Neural Network (RNN) well-suited for time series data. The model is trained on historical stock price data and can be used to forecast prices for a specified number of days into the future. This tool can be valuable for investors, traders, and financial analysts.

## Features
- Predicts stock prices using LSTM neural networks.
- Supports customizable time steps and prediction horizons.
- Visualizes historical and predicted stock prices.
- Easy-to-use interface for training and testing the model.
- Compatible with any stock ticker symbol (e.g., AAPL, TSLA, GOOGL).

## Installation
Follow these steps to set up the project on your local machine.

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-prediction-lstm.git
   cd stock-price-prediction-lstm
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download historical stock price data (e.g., from [Yahoo Finance](https://finance.yahoo.com/)) and place it in the `data/` directory. Alternatively, use the provided script to fetch data automatically.

## Usage
To train and use the LSTM model for stock price prediction:

1. Prepare the dataset:
   - If you have a CSV file with historical stock prices, place it in the `data/` directory.
   - Alternatively, use the `data_download.py` script to fetch data for a specific ticker symbol:
     ```bash
     python data_download.py --ticker AAPL --start_date 2010-01-01 --end_date 2023-01-01
     ```

2. Train the LSTM model:
   ```bash
   python train.py --ticker AAPL --epochs 50 --batch_size 32 --time_steps 60
   ```
   - `--ticker`: Stock ticker symbol (e.g., AAPL).
   - `--epochs`: Number of training epochs.
   - `--batch_size`: Batch size for training.
   - `--time_steps`: Number of time steps (historical data points) used for prediction.

3. Predict future stock prices:
   ```bash
   python predict.py --ticker AAPL --future_days 30
   ```
   - `--ticker`: Stock ticker symbol (e.g., AAPL).
   - `--future_days`: Number of days to predict into the future.

4. Visualize the results:
   - The script will generate a plot showing the historical and predicted stock prices.

## Dataset
The model can be trained on any historical stock price dataset. The dataset should include at least the following columns:
- `Date`: The date of the stock price.
- `Close`: The closing price of the stock.

Example dataset format:
| Date       | Open  | High  | Low   | Close | Volume   |
|------------|-------|-------|-------|-------|----------|
| 2023-01-01 | 130.0 | 132.0 | 129.0 | 131.0 | 12345678 |

## Model Architecture
The LSTM model consists of:
- Input layer with `time_steps` as the sequence length.
- Two LSTM layers with 50 units each.
- A Dense output layer with 1 unit (for predicting the stock price).

## Contributing
We welcome contributions! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request with a detailed description of your changes.

## Acknowledgments
- Thanks to the creators of TensorFlow and Keras for providing the tools to build and train neural networks.
- This project was inspired by the growing interest in using machine learning for financial forecasting.

## Contact
For questions or feedback, please reach out to:
- **Email** - (info.prashant248@gmail.com)
- **Project Repository** - (https://github.com/prashant24802/Stock-Market-Prediction-using-LSTM-Recurrent-Neural-Network)

