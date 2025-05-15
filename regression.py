import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Argumenty wejściowe
parser = argparse.ArgumentParser(description="Model regresyjny")
parser.add_argument("filename", type=str, help="Nazwa pliku")
args = parser.parse_args()

# Wczytywanie danych
data = np.loadtxt(args.filename)
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, activation='tanh'):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        if activation == 'tanh':
            self.activation = np.tanh
            self.activation_deriv = lambda x: 1 - np.tanh(x) ** 2
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            self.activation_deriv = lambda x: self.activation(x) * (1 - self.activation(x))
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_deriv = lambda x: (x > 0).astype(float)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, y, output, lr=0.01):
        m = X.shape[0]
        dz2 = output - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = dz2 @ self.W2.T * self.activation_deriv(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X, y, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, lr)
            if epoch % 200 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)


def evaluate(model, X, y, name=""):
    pred = model.predict(X)
    mse = np.mean((pred - y) ** 2)
    print(f"[{name}] MSE: {mse:.4f}")
    return mse


if __name__ == "__main__":
    # Sieć z aktywacją tanh
    print("\n--- Trening sieci z aktywacją tanh ---")
    model_tanh = NeuralNetwork(input_dim=X.shape[1], hidden_dim=10, output_dim=1, activation='tanh')
    model_tanh.train(X_train, y_train, epochs=1000, lr=0.05)
    mse_tanh = evaluate(model_tanh, X_test, y_test, name="TANH")

    # Sieć z aktywacją ReLU
    print("\n--- Trening sieci z aktywacją ReLU ---")
    model_relu = NeuralNetwork(input_dim=X.shape[1], hidden_dim=10, output_dim=1, activation='relu')
    model_relu.train(X_train, y_train, epochs=1000, lr=0.05)
    mse_relu = evaluate(model_relu, X_test, y_test, name="RELU")


    # Sieć z aktywacją SIGMOID
    print("\n--- Trening sieci z aktywacją SIGMOID ---")
    model_sigm = NeuralNetwork(input_dim=X.shape[1], hidden_dim=10, output_dim=1, activation='sigmoid')
    model_sigm.train(X_train, y_train, epochs=1000, lr=0.05)
    mse_sigm = evaluate(model_relu, X_test, y_test, name="SIGMOID")

    # Prosta ocena jakości dopasowania
    print("\n--- Ocena dopasowania ---")
    for name, mse in [("TANH", mse_tanh), ("RELU", mse_relu), ("SIGMOID", mse_sigm)]:
        if mse > 1.0:
            print(f"{name}: Zbyt małe dopasowanie (underfitting)")
        elif mse < 1e-3:
            print(f"{name}: Możliwe przeuczenie (overfitting)")
        else:
            print(f"{name}: Optymalne dopasowanie")