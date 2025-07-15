# 📈 Stock Market Direction Prediction Using Machine Learning

This repository presents a machine learning project focused on predicting the next-day direction of the stock market using a variety of technical indicators and three different ML algorithms. By classifying whether the market will go up or down, the models aim to provide actionable signals, which are then used in a simple trading strategy designed to outperform the market.

## 🔍 Project Overview

The goal of this project is to classify the next-day movement of a stock or index (up or down) based on historical price-derived features. To achieve this, we extract a set of widely used technical indicators from the price data, define a suitable target variable, and train three types of supervised learning models to make predictions.

Finally, we implement a rule-based trading strategy that uses the model outputs to make long/short decisions.

## Packages


```setup
pip install -r yfinance
```

## 📊 Technical Indicators

The following technical indicators are calculated and used as input features for the models:

* Simple Moving Average (SMA): $$\text{SMA}_ n (t) =  \frac {1} {n}  \sum_{i=0}^{n-1} P_{t-i}$$
   
* Exponential Moving Average (EMA): $\text{EMA}_ t = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_{t-1}, \quad \alpha = \frac{2}{n+1}$

* Relative Strength Index (RSI): $\text{RSI}_ t = 100 - \left( \frac{100}{1 + RS_t} \right), \quad RS_t = \frac{\text{Average Gain}}{\text{Average Loss}}$

* Bollinger Bands: $\text{Upper Band} = \text{SMA}_n + k \cdot \sigma_n $,
$\text{Lower Band} = \text{SMA}_n - k \cdot \sigma_n$

* Momentum: $\text{Momentum}_ n (t) = P_t - P_{t-n}$

* MACD (Moving Average Convergence Divergence): $\text{MACD}_ t = \text{EMA}_ {12}(t) - \text{EMA}_{26}(t)$

* Daily Return: $r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$

* Rolling Volatility (e.g., 5-day standard deviation): $\sigma_t = \sqrt{\frac{1}{n} \sum_{i=0}^{n-1} (r_{t-i} - \bar{r})^2}$

* Lagged Returns: $r_{t-1},\ r_{t-2},\ \ldots,\ r_{t-k}$


## 🤖 Machine Learning Models Used

We train and compare the performance of the following machine learning models:

**Logistic Regression:**  
We implement Logistic Regression as it is done in the lecture, but add lasso and ridge regularization to further improve the results and prevent oveerfitting.   
  
**Kernel Support Vector Machine (SVM):**  
For SVM we try linear, polynomial and rbf kernels with different parameters for the last two, to decide which model works best in a stock environment.
  
**Multilayer Neural Network:**  
We train a neural network with 2 hidden layers, and try to achieve better results than the other two models, while neglecting runtime, to test the limit of the technical indicators.  

## 💡 Trading Strategy

Using the predictions from the models, we implement a basic trading strategy:

Long position when the model predicts an upward move.
Short or neutral position when the model predicts a downward move.
The strategy’s performance is evaluated against a simple buy-and-hold benchmark to test whether the model can provide a predictive edge in real-world trading.

## Packages and Usage

For the code we use the courselib library, the packages we used in the lecture, yfinance for the stock data and seaborn for better visuals on the confusion matrices.

The code is run can be run from Top to bottom, but there is the possibility to skip either single plots, that you are not interested in, or even entire Models, if you just want to check a certain one.
