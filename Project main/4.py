import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from courselib.models.nn import Sigmoid
from courselib.models.svm import BinaryKernelSVM
import cvxopt
import time
import itertools
from courselib.utils.splits import train_test_split
from courselib.models.base import TrainableModel
from courselib.utils.metrics import binary_accuracy, mean_squared_error, mean_absolute_error, accuracy
from courselib.optimizers import GDOptimizer
from courselib.models.linear_models import LinearBinaryClassification, RidgeClassifier
from courselib.utils.normalization import min_max
from courselib.models.glm import LogisticRegression
from courselib.models.base import TrainableModel
from courselib.utils.preprocessing import labels_encoding

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



def normalize(x):
    return (x - np.mean(x, axis=0))/np.std(x,axis=0)

class ReLU:
    """ReLU activation function and its subgradient."""

    def __call__(self, x):
        return np.maximum(0, x)

    def grad(self, x):
        return (x > 0).astype(float)


class Linear:
    """Linear activation (identity function) and its derivative."""

    def __call__(self, x):
        return x

    def grad(self, x):
        return np.ones_like(x)



class MSE:
    """Quadratic (L2) loss and its gradient."""

    def __call__(self, Y_pred, Y_true):
        return 0.5 * np.mean((Y_pred - Y_true) ** 2)

    def grad(self, Y_pred, Y_true):
        return (Y_pred - Y_true) / Y_pred.shape[0]


class DenseLayer:
    """
    Fully-connected (dense) layer with activation.

    Parameters:
    - num_in: number of input neurons
    - num_out: number of output neurons
    - activation: activation class (e.g., ReLU or Linear)
    - layer_name: unique name to identify the layer (used for parameter keys)
    """

    def __init__(self, num_in, num_out, activation=ReLU, layer_name=None):
        self.num_in = num_in
        self.num_out = num_out
        self.activation = activation()
        self.name = layer_name or f"Dense_{num_in}_{num_out}"

        # He initialization (good for ReLU)
        self.W = np.random.normal(loc=0.0, scale=np.sqrt(2. / num_in), size=(num_in, num_out))
        self.b = np.zeros((1, num_out))

    def __call__(self, X):
        """
        Forward pass through the layer.

        Parameters:
        - X: input matrix of shape (N, num_in)

        Returns:
        - z: pre-activation (N, num_out)
        - x: post-activation (N, num_out)
        """
        z = X @ self.W + self.b
        x = self.activation(z)
        return z, x

    def compute_delta(self, z, W_next, delta_next):
        """
        Compute backpropagated error for this layer.

        Parameters:
        - z: pre-activation from this layer (N, num_out)
        - W_next: weights of the next layer (num_out, next_layer_size)
        - delta_next: delta from the next layer (N, next_layer_size)

        Returns:
        - delta for this layer (N, num_out)
        """
        return (delta_next @ W_next.T) * self.activation.grad(z)

    def loss_grad(self, X_prev, delta):
        """
        Compute gradients w.r.t. weights and biases.

        Parameters:
        - X_prev: input to this layer (N, num_in)
        - delta: error signal from this layer (N, num_out)

        Returns:
        - Dictionary with gradients for weights and biases
        """
        w_grad = np.mean(delta[:, :, None] * X_prev[:, None, :], axis=0).T  # (num_in, num_out)
        b_grad = np.mean(delta, axis=0, keepdims=True)  # (1, num_out)

        return {f'{self.name}_W': w_grad, f'{self.name}_b': b_grad}

    def _get_params(self):
        """Return a dictionary of parameters for this layer."""
        return {f'{self.name}_W': self.W, f'{self.name}_b': self.b}


class MLP(TrainableModel):
    def __init__(self, widths, optimizer, activation=ReLU, output_activation=Linear, loss=MSE):
        """
        Initializes a multi-layer perceptron (MLP) using a sequence of DenseLayers.

        Parameters:
        - widths: list of layer sizes, including input and output dimensions
        - optimizer: optimizer instance (must support `update(params, grads)`)
        - activation: activation class for hidden layers
        - output_activation: activation class for the output layer
        - loss: loss function class
        """
        self.optimizer = optimizer
        self.widths = widths
        self.loss = loss()

        # Build hidden layers
        self.layers = [
            DenseLayer(widths[i], widths[i + 1], activation=activation, layer_name=f"layer_{i}")
            for i in range(len(widths) - 2)
        ]

        # Output layer
        self.layers.append(
            DenseLayer(widths[-2], widths[-1], activation=output_activation, layer_name=f"layer_{len(widths) - 2}")
        )

    def decision_function(self, X):
        """Applies all layers to compute the raw network output."""
        out = X
        for layer in self.layers:
            _, out = layer(out)
        return out

    def __call__(self, X):
        return np.argmax(self.decision_function(X), axis=-1)

    def forward_pass(self, X):
        """Computes pre-activations and activations at each layer."""
        x_l = [X]
        z_l = [X]
        for layer in self.layers:
            z, x = layer(x_l[-1])
            z_l.append(z)
            x_l.append(x)
        return z_l, x_l

    def backward_pass(self, X, Y, z_l, x_l):
        """Computes layer-wise gradients using backpropagation."""
        delta = self.loss.grad(x_l[-1], Y) * self.layers[-1].activation.grad(z_l[-1])
        deltas = [delta]

        for i in reversed(range(len(self.layers) - 1)):
            delta = self.layers[i].compute_delta(z_l[i + 1], self.layers[i + 1].W, deltas[-1])
            deltas.append(delta)

        return deltas[::-1]

    def loss_grad(self, X, Y):
        """Returns gradients of the loss w.r.t. all layer parameters."""
        z_l, x_l = self.forward_pass(X)
        delta_l = self.backward_pass(X, Y, z_l, x_l)

        grads = {}
        for i, layer in enumerate(self.layers):
            grads.update(layer.loss_grad(x_l[i], delta_l[i]))
        return grads

    def _get_params(self):
        """Returns a dictionary of all layer parameters."""
        params = {}
        for layer in self.layers:
            params.update(layer._get_params())
        return params




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

    Y, Y_train, Y_test = labels_encoding(Y), labels_encoding(Y_train), labels_encoding(Y_test)

    metrics_dict = {
        "accuracy": accuracy,
        "loss": mean_squared_error
    }


    optimizer = GDOptimizer(learning_rate=1)

    model = MLP(widths=[14, 64,32, 2], optimizer=optimizer, activation=ReLU, output_activation=Linear, loss=MSE)

    metrics_history = model.fit(X_train, Y_train, num_epochs=15000, batch_size=len(X_train), compute_metrics=True,
                                metrics_dict=metrics_dict)

    fig, ax = plt.subplots()

    ax.plot(range(len(metrics_history['loss'])), metrics_history['loss'])
    ax.set_ylabel('Loss value')

    ax2 = ax.twinx()
    ax2.plot(range(len(metrics_history['accuracy'])), metrics_history['accuracy'], color='orange')
    ax2.set_ylabel('Accuracy')

    ax.set_xlabel('Epoch')

    plt.title('Learning curve')
    plt.grid()
    plt.show()

    print(f'The final train accuracy: {round(metrics_history["accuracy"][-1], 1)}%')
    print(f'Test accuracy: {round(accuracy(model.decision_function(X_test), Y_test), 1)}%')