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


# Model 1: Liniowy model Y = w0 + w1 * X
def add_bias(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def gradient_descent(X, y, lr=0.01, n_iters=1000):
    m, n = X.shape
    w = np.zeros((n, 1))
    for _ in range(n_iters):
        y_pred = X @ w
        error = y_pred - y
        grad = (2 / m) * X.T @ error
        w -= lr * grad
    return w


X_train_bias = add_bias(X_train)
X_test_bias = add_bias(X_test)

w1 = gradient_descent(X_train_bias, y_train)

# Predykcja
y_pred_train1 = X_train_bias @ w1
y_pred_test1 = X_test_bias @ w1

# Walidacja Modelu 1
mse_train1 = mean_squared_error(y_train, y_pred_train1)
mse_test1 = mean_squared_error(y_test, y_pred_test1)

print("\nModel 1 - Liniowy:")
print(f"MSE train: {mse_train1:.4f}")
print(f"MSE test: {mse_test1:.4f}")

# Model 2: Rozszerzony model nieliniowy Y = w0 + w1*X + w2*X^2
X2_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train, X_train ** 2))
X2_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test, X_test ** 2))

w2 = gradient_descent(X2_train, y_train)

y_pred_train2 = X2_train @ w2
y_pred_test2 = X2_test @ w2

# Walidacja Modelu 2
mse_train2 = mean_squared_error(y_train, y_pred_train2)
mse_test2 = mean_squared_error(y_test, y_pred_test2)

print("\nModel 2 - Kwadratowy:")
print(f"MSE train: {mse_train2:.4f}")
print(f"MSE test: {mse_test2:.4f}")

# Porównanie
print("\nPorównanie modeli:")
print(f"Model 1 (test MSE): {mse_test1:.4f}")
print(f"Model 2 (test MSE): {mse_test2:.4f}")
if mse_test1 < mse_test2:
    print("Lepszy wynik testowy ma Model 1 (liniowy)")
else:
    print("Lepszy wynik testowy ma Model 2 (kwadratowy)")

# Wizualizacja
plt.scatter(X, y, label="Dane", alpha=0.5)
x_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_plot_bias = add_bias(x_plot)
x_plot_quad = np.hstack((np.ones((x_plot.shape[0], 1)), x_plot, x_plot ** 2))

plt.plot(x_plot, x_plot_bias @ w1, label="Model 1 (liniowy)", color='red')
plt.plot(x_plot, x_plot_quad @ w2, label="Model 2 (kwadratowy)", color='green')
plt.legend()
plt.title("Porównanie modeli regresyjnych")
plt.show()
