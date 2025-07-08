import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import normal
from fastjsonschema.ref_resolver import normalize

from courselib.utils.splits import train_test_split
from courselib.models.base import TrainableModel
from courselib.utils.metrics import binary_accuracy, mean_squared_error, mean_absolute_error
from courselib.optimizers import GDOptimizer
from courselib.models.linear_models import LinearBinaryClassification, RidgeClassifier, LinearRegression
from courselib.utils.normalization import min_max
from courselib.models.glm import LogisticRegression
from courselib.models.base import TrainableModel

def fetch_data(ticker='SPY', start='2015-01-01', end='2023-12-31'):
    data = yf.download(ticker, start, end)
    df = pd.DataFrame(data)
    df.columns = df.columns.droplevel(1)
    return df



def compute_technical_indicators(data,amount,window):
    df = data.copy()

    df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()

    df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = delta.clip(upper=0).abs()
    avg_gain = up.ewm(window, adjust=False).mean()
    avg_loss = down.ewm(window, adjust=False).mean()
    rs = avg_gain / avg_loss
    df[f'RSI_{window}'] = 100 - (100 / (1 + rs))

    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = sma + 2 * std
    df['Bollinger_Lower'] = sma - 2 * std

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Daily_Return'] = df['Close'].pct_change()


    df['Rolling_5d_Std'] = df['Daily_Return'].rolling(window=5).std()

    for i in range(amount):
        df[f'Lag_{i+1}'] = df['Daily_Return'].shift(i)

    # Momentum
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    return df


class LinearSVM(TrainableModel):

    def __init__(self, w, b, optimizer, C=10.):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)
        self.C = C

    def loss_grad(self, X, y):
        # Compute raw model output
        output = self.decision_function(X)

        # Identify margin violations: where 1 - y*h(x) > 0
        mask = (1 - y * output) > 0
        y_masked = y[mask]
        X_masked = X[mask]

        # Compute gradients
        grad_w = 2 * self.w - self.C * np.mean(y_masked[:, None] * X_masked, axis=0) if len(
            y_masked) > 0 else 2 * self.w
        grad_b = - self.C * np.mean(y_masked) if len(y_masked) > 0 else 0.0

        return {"w": grad_w, "b": grad_b}

    def decision_function(self, X):
        return X @ self.w + self.b

    def _get_params(self):
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

def normalize(x):
    return (x - np.mean(x, axis=0))/np.std(x,axis=0)



# Usage
if __name__ == '__main__':
    df = fetch_data('SPY', '2015-01-01', '2023-12-31')
    df['Trend'] = df['Close'].rolling(window=5).mean().shift(-1) > df['Close']
    df['Target'] = df['Trend'].astype(int)
    df = compute_technical_indicators(df, 3, 14)

    features = ['Lag_1', 'Lag_2', 'Lag_3','Target','Close']

    df = df[features]
    df = df.reset_index()
    df.dropna(inplace=True)
    print(df.corr()['Target'].sort_values(ascending=False))
    X, Y, X_train, Y_train, X_test, Y_test = train_test_split(df.iloc[:, df.columns != 'Date'],
                                                              training_data_fraction=0.8,
                                                              return_numpy=True,
                                                              class_column_name='Target')

    X = min_max(X)
    X_train = min_max(X_train)
    X_test = min_max(X_test)

    w = [0] * X.shape[1]
    b = 0
    optimizer = GDOptimizer(learning_rate=1e-2)

    accuracy = lambda y_true, y_pred: binary_accuracy(y_true, y_pred, class_labels=[0, 1])
    metrics_dict = {'accuracy': accuracy,
                    'MSE': mean_squared_error}

    model = LinearRegression(w, b, optimizer)
    metrics_history = model.fit(X_train, Y_train, num_epochs=20000, batch_size=len(X_train), compute_metrics=True,
                                metrics_dict=metrics_dict)

    fig, ax = plt.subplots()

    ax.plot(range(len(metrics_history['MSE'])), metrics_history['MSE'])
    ax.set_ylabel('Loss value')

    ax2 = ax.twinx()
    ax2.plot(range(len(metrics_history['accuracy'])), metrics_history['accuracy'], color='orange')
    ax2.set_ylabel('Accuracy')

    ax.set_xlabel('Epoch')

    plt.title('Learning curve')
    plt.grid()

    print(f'The final train accuracy: {metrics_history["accuracy"][-1]}%')
    print(f'Test accuracy: {accuracy(model.decision_function(X_test), Y_test)}%')

    plt.show()