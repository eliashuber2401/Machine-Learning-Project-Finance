# 📈 Stock Market Direction Prediction Using Machine Learning

This repository presents a machine learning project focused on predicting the next-day direction of the stock market using a variety of technical indicators and three different ML algorithms. By classifying whether the market will go up or down, the models aim to provide actionable signals, which are then used in a simple trading strategy designed to outperform the market.

## 🔍 Project Overview

The goal of this project is to classify the next-day movement of a stock or index (up or down) based on historical price-derived features. To achieve this, we extract a set of widely used technical indicators from the price data, define a suitable target variable, and train three types of supervised learning models to make predictions.

Finally, we implement a rule-based trading strategy that uses the model outputs to make long/short decisions.

## 📊 Technical Indicators

The following technical indicators are calculated and used as input features for the models:

* Simple Moving Average (SMA):

* Exponential Moving Average (EMA):

* Relative Strength Index (RSI):

* Bollinger Bands:

* Momentun:

* MACD (Moving Average Convergence Divergence):

* Daily Return:

* Rolling Volatility (e.g., 5-day STD):
Standard deviation of daily returns over a rolling window (e.g., 5 days) to measure market risk.
* Lagged Returns:

🎯 Target Variable

The classification target is defined as follows:

1 (Positive) if the price increases the next day.
0 or -1 (Negative) if the price stays the same or decreases.
Depending on the specific model, we may use a binary (0/1) or ternary (-1/0/1) classification format.

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
