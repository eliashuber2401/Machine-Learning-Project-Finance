import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from courselib.models.svm import BinaryKernelSVM
import cvxopt
import time
import itertools
from courselib.utils.splits import train_test_split
from courselib.models.base import TrainableModel
from courselib.utils.metrics import binary_accuracy, mean_squared_error, mean_absolute_error
from courselib.optimizers import GDOptimizer
from courselib.models.linear_models import LinearBinaryClassification, RidgeClassifier
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

class SpectrumKernel:
    def __init__(self, k, X):
        self.k = k
        vocab = {c for seq in X for c in seq}
        self.kmers = [''.join(kmer) for kmer in itertools.product(vocab, repeat=k)]

    def feature_map(self, X):
        Phi = np.zeros((len(X), len(self.kmers)))
        for i, seq in enumerate(X):
            for j, kmer in enumerate(self.kmers):
                count = seq.count(kmer)
                Phi[i, j] = count

        return Phi


    def __call__(self, X1, X2):
        Phi1 = self.feature_map(X1)
        Phi2 = self.feature_map(X2)
        return Phi1 @ Phi2.T


class Kernel:
    def _check_shapes(self, X1, X2):
        if X1.shape[-1] != X2.shape[-1]:
            raise ValueError("Inputs must have the same number of features (last dimension).")

        d = X1.shape[-1]
        return X1.reshape(-1, d), X2.reshape(-1, d)


class LinearKernel(Kernel):
    def __call__(self, X1, X2):
        X1_flat, X2_flat = self._check_shapes(X1, X2)
        return (X1_flat @ X2_flat.T).reshape(X1.shape[:-1] + X2.shape[:-1])


class PolynomialKernel(Kernel):
    def __init__(self, degree=2, intercept=1):
        self.degree = degree
        self.intercept = intercept

    def __call__(self, X1, X2):
        X1_flat, X2_flat = self._check_shapes(X1, X2)
        prod = (X1_flat @ X2_flat.T).reshape(X1.shape[:-1] + X2.shape[:-1])
        return np.power(prod + self.intercept, self.degree)


class RBFKernel(Kernel):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, X1, X2):
        X1_flat, X2_flat = self._check_shapes(X1, X2)
        diff = np.linalg.norm(
            X1_flat[:, np.newaxis, :] - X2_flat[np.newaxis, :, :],
            axis=-1
        ).reshape(X1.shape[:-1] + X2.shape[:-1])
        return np.exp(-diff ** 2 / (2 * self.sigma ** 2))


# Usage
if __name__ == '__main__':
    df = fetch_data('SPY', '2015-01-01', '2023-12-31')
    df['Trend'] = df['Close'].rolling(window=5).mean().shift(-1) > df['Close']
    df['Target'] = np.where(df['Trend'], 1, -1)
    df = compute_technical_indicators(df, 3, 14)

    features = ['SMA_14', 'EMA_14', 'RSI_14', 'Bollinger_Upper', 'Bollinger_Lower',
                'MACD', 'MACD_Signal', 'Daily_Return', 'Rolling_5d_Std',
                'Lag_1', 'Lag_2', 'Lag_3', 'Momentum_10','Target','Close']

    df = df[features]
    df = df.reset_index()
    df.dropna(inplace=True)

    X, Y, X_train, Y_train, X_test, Y_test = train_test_split(df.iloc[:, df.columns != 'Date'],
                                                              training_data_fraction=0.8,
                                                              return_numpy=True,
                                                              shuffle=False,
                                                              class_column_name='Target')

    X = min_max(X)
    X_train = min_max(X_train)
    X_test = min_max(X_test)

    kernels = ['linear', 'polynomial', 'rbf']
    for i in range(10):
        start = time.time()
        svm = BinaryKernelSVM(kernel='polynomial', degree=i)
        svm.fit(X_train, Y_train)
        end = time.time()

        test_acc = binary_accuracy(svm(X_test), Y_test)

        train_acc = binary_accuracy(svm(X_train), Y_train)

        print(f'Test accuracy degree {i}: {test_acc:.4f}, Train accuracy: {train_acc:.4f}')

        train_time = end - start

        print(f"⏱️ Train time: {train_time:.4f} seconds")

    sigma_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    for i in sigma_values:
        start = time.time()
        svm = BinaryKernelSVM(kernel='rbf', sigma=i)
        svm.fit(X_train, Y_train)
        end = time.time()

        test_acc = binary_accuracy(svm(X_test), Y_test)

        train_acc = binary_accuracy(svm(X_train), Y_train)

        print(f'Test accuracy sigma {i}: {test_acc:.4f}, Train accuracy: {train_acc:.4f}')

        train_time = end - start

        print(f"⏱️ Train time: {train_time:.4f} seconds")