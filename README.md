# 📈 Stock Market Direction Prediction Using Machine Learning

This repository presents a machine learning project focused on predicting the next-day direction of the stock market using a variety of technical indicators and three different ML algorithms. By classifying whether the market will go up or down, the models aim to provide actionable signals, which are then used in a simple trading strategy designed to outperform the market.

## 🔍 Project Overview

The goal of this project is to classify the next-day movement of a stock or index (up or down) based on historical price-derived features. To achieve this, we extract a set of widely used technical indicators from the price data, define a suitable target variable, and train three types of supervised learning models to make predictions.

Finally, we implement a rule-based trading strategy that uses the model outputs to make long/short decisions.

## 📊 Technical Indicators

The following technical indicators are calculated and used as input features for the models:

* Simple Moving Average (SMA):$$\text{SMA}_n(t) = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}$$
* 
* Exponential Moving Average (EMA):
$\[
\text{EMA}_t = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_{t-1}, \quad \alpha = \frac{2}{n+1}
\]$

* Relative Strength Index (RSI):$\text{RSI}_t = 100 - \left( \frac{100}{1 + RS_t} \right), \quad RS_t = \frac{\text{Average Gain}}{\text{Average Loss}}$

* Bollinger Bands:
$\[
\text{Upper Band} = \text{SMA}_n + k \cdot \sigma_n, \quad
\text{Lower Band} = \text{SMA}_n - k \cdot \sigma_n
\]$

* Momentum:
$\[
\text{Momentum}_n(t) = P_t - P_{t-n}
\]$

* MACD (Moving Average Convergence Divergence):
$\[
\text{MACD}_t = \text{EMA}_{12}(t) - \text{EMA}_{26}(t)
\]$

* Daily Return:
$\[
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
\]$

* Rolling Volatility (e.g., 5-day standard deviation):
$\[
\sigma_t = \sqrt{\frac{1}{n} \sum_{i=0}^{n-1} (r_{t-i} - \bar{r})^2}
\]$

* Lagged Returns:
$\[
r_{t-1},\ r_{t-2},\ \ldots,\ r_{t-k}
\]$


## 🤖 Machine Learning Models Used

We train and compare the performance of the following machine learning models:

Logistic Regression
A linear model for binary classification that estimates the probability of an upward move.
Kernel Support Vector Machine (SVM)
A non-linear classifier that finds an optimal decision boundary using a kernel function.
Multilayer Neural Network
A feedforward neural network with one or more hidden layers, capable of learning complex patterns in the data.
## 💡 Trading Strategy

Using the predictions from the models, we implement a basic trading strategy:

Long position when the model predicts an upward move.
Short or neutral position when the model predicts a downward move.
The strategy’s performance is evaluated against a simple buy-and-hold benchmark to test whether the model can provide a predictive edge in real-world trading.
