# 📈 Stock Market Direction Prediction Using Machine Learning

This repository presents a machine learning project focused on predicting the next-day direction of the stock market using a variety of technical indicators and three different ML algorithms. By classifying whether the market will go up or down, the models aim to provide actionable signals, which are then used in a simple trading strategy designed to outperform the market.

## 🔍 Project Overview

The goal of this project is to classify the next-day movement of a stock or index (up or down) based on historical price-derived features. To achieve this, we extract a set of widely used technical indicators from the price data, define a suitable target variable, and train three types of supervised learning models to make predictions.

Finally, we implement a rule-based trading strategy that uses the model outputs to make long/short decisions.

## 📊 Technical Indicators

The following technical indicators are calculated and used as input features for the models:

* Simple Moving Average (SMA):
SMA
n
=
1
n
∑
i
=
0
n
−
1
P
t
−
i
SMA 
n
​	
 = 
n
1
​	
  
i=0
∑
n−1
​	
 P 
t−i
​	
Average of the closing prices over the last n days.
* Exponential Moving Average (EMA):
EMA
t
=
α
⋅
P
t
+
(
1
−
α
)
⋅
EMA
t
−
1
EMA 
t
​	
 =α⋅P 
t
​	
 +(1−α)⋅EMA 
t−1
​	
 
* More weight is given to recent prices, where 
α
=
2
n
+
1
α= 
n+1
2
​	

* Relative Strength Index (RSI):
RSI
=
100
−
(
100
1
+
R
S
)
RSI=100−( 
1+RS
100
​	
 )
where 
R
S
=
avg gain
avg loss
RS= 
avg loss
avg gain
​	
 , calculated over 14 days. RSI measures the speed and change of price movements.
* Bollinger Bands:
Consist of an upper and lower band based on standard deviation from a moving average:
Upper Band
=
SMA
+
k
⋅
σ
,
Lower Band
=
SMA
−
k
⋅
σ
Upper Band=SMA+k⋅σ,Lower Band=SMA−k⋅σ
Indicates volatility and potential overbought/oversold conditions.
* Momentum:
Momentum
n
=
P
t
−
P
t
−
n
Momentum 
n
​	
 =P 
t
​	
 −P 
t−n
​	
 
* Measures the velocity of price changes.
MACD (Moving Average Convergence Divergence):
MACD
=
EMA
12
−
EMA
26
MACD=EMA 
12
​	
 −EMA 
26
​	
 
Often paired with a signal line (9-day EMA of MACD) to identify potential buy/sell points.
* Daily Return:
r
t
=
P
t
−
P
t
−
1
P
t
−
1
r 
t
​	
 = 
P 
t−1
​	
 
P 
t
​	
 −P 
t−1
​	
 
​	
 
* Rolling Volatility (e.g., 5-day STD):
Standard deviation of daily returns over a rolling window (e.g., 5 days) to measure market risk.
* Lagged Returns:
Lagged versions of the daily return (e.g., 
r
t
−
1
,
r
t
−
2
,
…
r 
t−1
​	
 ,r 
t−2
​	
 ,…) are used to capture temporal dependencies.
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
