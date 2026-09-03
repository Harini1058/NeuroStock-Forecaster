
# 📈 Developing a Recurrent Neural Network Model for Stock Prediction

## 👩‍💻 Project Information

**Name:** Harini G

**Register Number:** 212225230091

**Program:** B.Tech Artificial Intelligence and Data Science

**Project:** Developing a Recurrent Neural Network Model for Stock Prediction

**Framework:** PyTorch

---

## 📌 Project Overview

This project focuses on developing a **Recurrent Neural Network (RNN)** model using **PyTorch** to predict future stock prices based on historical stock-market data.

The model uses the **Closing Price (`Close`)** of the stock as the primary input. Historical prices are transformed into sequential data using a **60-day time window**, allowing the RNN to learn temporal patterns and relationships in stock-price movements.

The project includes:

* Loading historical stock-price data
* Data preprocessing and normalization
* Creating time-series sequences
* Building an RNN architecture
* Training the model using PyTorch
* Evaluating predictions on test data
* Converting predictions back to original price values
* Visualizing training loss
* Comparing actual and predicted stock prices

---

## 🎯 Objectives

The main objectives of this project are:

1. To understand stock-price time-series data.
2. To preprocess and normalize stock-price values.
3. To create sequential training data using historical prices.
4. To develop an RNN model using PyTorch.
5. To train the model for stock-price prediction.
6. To evaluate the model using prediction error metrics.
7. To visualize actual and predicted stock prices.

---

## 🗂️ Dataset

The project uses two CSV files:

```text
trainset.csv
testset.csv
```

Both datasets contain the following columns:

| Column    | Description                  |
| --------- | ---------------------------- |
| Date      | Trading date                 |
| Open      | Opening stock price          |
| High      | Highest price during the day |
| Low       | Lowest price during the day  |
| Close     | Closing stock price          |
| Adj Close | Adjusted closing price       |
| Volume    | Trading volume               |

### Dataset Size

* **Training data:** 1,259 records
* **Testing data:** 125 records
* **Input feature:** `Close`

The model specifically uses the **Close** column for prediction.

---

## 🔄 Data Preprocessing

The closing-price values are extracted from the training and testing datasets.

A **MinMaxScaler** is used to normalize the data:

```text
Original Prices
      ↓
Min-Max Scaling
      ↓
Normalized Prices
      ↓
Sequence Creation
```

The scaler is fitted only on the training data to avoid data leakage.

---

## ⏳ Sequence Creation

The model uses a **sequence length of 60 days**.

This means the model uses the previous **60 stock prices** to predict the next stock price.

For example:

```text
Day 1 ─┐
Day 2  │
Day 3  │
 ...   ├──→ RNN ──→ Predicted Price
Day 59 │
Day 60 ┘
```

The sequence structure is:

```text
Input  : Previous 60 closing prices
Output : Next closing price
```

With the given datasets:

* Training sequences: **1,199**
* Testing sequences: **65**

---

## 🧠 RNN Architecture

The project uses a PyTorch **Recurrent Neural Network** with the following architecture:

```text
Input
  │
  │  60 time steps × 1 feature
  ↓
┌──────────────────────┐
│      RNN Layer       │
│ Hidden Size = 64     │
│ Number of Layers = 2 │
└──────────────────────┘
  │
  │ Last Time Step
  ↓
┌──────────────────────┐
│   Fully Connected    │
│       64 → 1         │
└──────────────────────┘
  │
  ↓
Predicted Stock Price
```

### Model Parameters

| Parameter            | Value |
| -------------------- | ----: |
| Input Size           |     1 |
| Hidden Size          |    64 |
| Number of RNN Layers |     2 |
| Output Size          |     1 |
| Sequence Length      |    60 |
| Batch Size           |    64 |

The RNN processes the historical sequence and uses the output from the **last time step** for predicting the next stock price.

---

## ⚙️ Training Configuration

The model is trained using the following configuration:

| Configuration | Value                            |
| ------------- | -------------------------------- |
| Framework     | PyTorch                          |
| Loss Function | Mean Squared Error (MSE)         |
| Optimizer     | Adam                             |
| Learning Rate | 0.001                            |
| Batch Size    | 64                               |
| Epochs        | 20                               |
| Device        | CUDA if available, otherwise CPU |

### Training Process

```text
Training Data
      ↓
Create 60-Day Sequences
      ↓
Convert to PyTorch Tensors
      ↓
DataLoader
      ↓
RNN Model
      ↓
Calculate MSE Loss
      ↓
Backpropagation
      ↓
Adam Optimizer
      ↓
Update Model Weights
```

---

## 📉 Training Loss Visualization

The training loss is recorded after every epoch and plotted to observe how the model learns over time.

The graph helps determine whether the model's error decreases during training.

```text
Loss
 │\
 │ \
 │  \
 │   \
 │    \____
 │
 └──────────────────→ Epochs
```

The actual training-loss values will depend on the model execution.

---

## 🧪 Model Evaluation

After training, the model is evaluated using the test dataset.

The predicted values are converted from normalized values back to their original stock-price scale using:

```python
scaler.inverse_transform()
```

This allows the predictions to be compared directly with the actual stock prices.

---

## 📊 Evaluation Metrics

The project calculates:

### Mean Squared Error (MSE)

MSE measures the average squared difference between actual and predicted prices.

```text
MSE = Average[(Actual − Predicted)²]
```

### Root Mean Squared Error (RMSE)

RMSE is calculated as:

```text
RMSE = √MSE
```

A lower RMSE indicates that the predicted prices are closer to the actual prices.

---

## 📈 Actual vs Predicted Stock Prices

The project generates a graph comparing:

* **Actual stock prices**
* **Predicted stock prices**

Conceptually:

```text
Price
 │       Actual
 │      /\/\/\
 │  ___/      \__
 │    Predicted
 │   /\/\/\/\
 │__/        \___
 └──────────────────→ Time
```

This visualization helps evaluate how closely the RNN follows the actual stock-price movement.

---

## 🖥️ Final Prediction

After testing, the program displays:

```text
Final Predicted Price : <value>
Final Actual Price    : <value>
```

These values represent the model's prediction and the corresponding actual closing price from the test sequence.

---

## 📁 Project Structure

```text
Stock-Prediction-RNN/
│
├── trainset.csv
├── testset.csv
├── stock_prediction_rnn.py
├── README.md
└── outputs/
    ├── training_loss.png
    └── actual_vs_predicted.png
```

---

## 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**

### Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
```

---

## 💻 Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Navigate to the project folder:

```bash
cd Stock-Prediction-RNN
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn torch
```

---

## ▶️ How to Run

Make sure the following files are available in the project directory:

```text
trainset.csv
testset.csv
```

Then run:

```bash
python stock_prediction_rnn.py
```

The program will:

1. Load the datasets.
2. Extract closing prices.
3. Normalize the data.
4. Create 60-day sequences.
5. Build the RNN model.
6. Train the model for 20 epochs.
7. Plot the training loss.
8. Generate test predictions.
9. Calculate MSE and RMSE.
10. Plot actual vs predicted prices.
11. Display the final predicted and actual prices.

---

## 🚀 Key Features

### 🔹 Time-Series Prediction

Uses historical stock prices to predict future values.

### 🔹 RNN-Based Model

Uses recurrent neural networks to capture sequential dependencies.

### 🔹 Data Normalization

Min-Max scaling improves model training by keeping input values within a consistent range.

### 🔹 GPU Support

Automatically uses CUDA when a compatible GPU is available.

### 🔹 Visualization

Provides training-loss and actual-vs-predicted graphs.

### 🔹 Performance Evaluation

Uses MSE and RMSE to measure prediction error.

---

## 📌 Advantages

* Suitable for sequential and time-series data.
* Simple and efficient RNN architecture.
* Can utilize GPU acceleration.
* Uses historical patterns for prediction.
* Provides visual evaluation of model performance.

---

## ⚠️ Limitations

Stock prices are influenced by many external factors that are not included in this model.

The current implementation:

* Uses only the `Close` price.
* Does not consider news or market sentiment.
* Does not use technical indicators.
* Does not include economic factors.
* Cannot guarantee accurate future stock-price predictions.

Therefore, the model should be considered an **educational machine-learning project**, not a financial decision-making system.

---

## 🔮 Future Enhancements

The project can be improved by:

* Using **LSTM** or **GRU** networks.
* Adding Open, High, Low, and Volume features.
* Including technical indicators such as RSI and MACD.
* Applying dropout for regularization.
* Performing hyperparameter tuning.
* Increasing the training dataset.
* Comparing RNN, LSTM, GRU, and Transformer models.
* Adding stock-market sentiment analysis.
* Developing an interactive prediction dashboard.

---

## 📚 Learning Outcomes

Through this project, the following concepts are demonstrated:

* Time-series data preprocessing
* Min-Max normalization
* Sequence generation
* Recurrent Neural Networks
* PyTorch model development
* DataLoader and batching
* Loss functions
* Backpropagation
* Adam optimization
* Model evaluation
* Data visualization
* Stock-price prediction

---

## 👩‍💻 Author

**Harini G**
**B.Tech Artificial Intelligence and Data Science**
**Register Number:** 212225230091
**Slot:** 26OD1143

---

## ⭐ Conclusion

This project demonstrates how a **Recurrent Neural Network can be used to learn patterns from historical stock-price sequences and generate predictions for future prices**.

By combining time-series preprocessing, PyTorch-based RNN architecture, model training, evaluation metrics, and visualization, the project provides a practical introduction to applying deep learning techniques to financial time-series data.

> **“Learn from the past to predict the future.”**
