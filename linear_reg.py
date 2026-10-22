import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
