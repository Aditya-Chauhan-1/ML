import numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(0)
X = np.linspace(-3, 3, 120).reshape(-1, 1)
y = 1.5*X**2 - 2*X + 5 + rng.normal(0, 1.0, size=X.shape)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_tr_p = poly.fit_transform(X_tr); X_te_p = poly.transform(X_te)

model = LinearRegression().fit(X_tr_p, y_tr)
y_pred = model.predict(X_te_p)
print("RMSE:", mean_squared_error(y_te, y_pred, squared=False), "R2:", r2_score(y_te, y_pred))

# visualize
X_full_p = poly.transform(X)
plt.scatter(X, y, s=10)
plt.plot(X, model.predict(X_full_p))
plt.title("Polynomial Regression (deg=2)"); plt.show()
