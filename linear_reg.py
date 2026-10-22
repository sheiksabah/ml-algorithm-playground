import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

X = np.array([ -3.0, -2.5, -2.0, -1.5, -1.0,
     0.0,  1.0,  1.5,  2.0,  2.5, 2.7])

y = np.array([ 15.5, 12.9, 9.5, 6.2, 5.8,
     5.5, 7.1, 9.7, 13.5, 18.4, 21.4])

X = X.reshape(-1, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Dataset for Linear Regression")
plt.show()

model = LinearRegression()
model.fit(X, y)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

w0 = model.intercept_
w1 = model.coef_[0]

print("Coefficient vector:", [float(w0), float(w1)])
print(f"Learned equation: y = {w0:.4f} + ({w1:.4f})x")

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
print("Training RMSE:", rmse)

plt.scatter(X, y, label="Actual data")
plt.plot(X, y_pred, color="red", label="Regression line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()
plt.show()

df = pd.read_csv("GasProperties.csv")

print(df.head())
print(df.columns)
print(df.shape)

X_data = df[["T", "P", "TC", "SV"]].to_numpy()
y_data = df["Idx"].to_numpy()

print("X_data shape:", X_data.shape)
print("y_data shape:", y_data.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

print("X_train_bias shape:", X_train_bias.shape)
print("X_test_bias shape:", X_test_bias.shape)

XtX = X_train_bias.T @ X_train_bias
Xty = X_train_bias.T @ y_train

w_hat = np.linalg.inv(XtX) @ Xty

print("Coefficient vector:", w_hat)

y_train_pred = X_train_bias @ w_hat
y_test_pred = X_test_bias @ w_hat

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Training RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)
