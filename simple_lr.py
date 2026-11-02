import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Create synthetic data: y = 3.5x + 7 + noise
rng = np.random.default_rng(42)
n = 300
X = rng.uniform(0, 10, size=(n, 1))                 # feature column[1,2,3,4,5]
noise = rng.normal(0, 2, size=(n, 1))               # random noise
y = (3.5 * X +7+ noise).ravel()                   # make target 1-D[1.2.3..4.5]

# 2) Train/Test split
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42
)

# 3) Train model
model = LinearRegression()
model.fit(X_train, y_train)
print(X_test)
# 4) Predictions
y_pred = model.predict([[1.79]])
print(y_test)
print(y_pred)
# 5) Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE using sqrt
r2 = r2_score(y_test, y_pred)

print(f"Slope: {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# 6) Plot regression line
plt.scatter(X, y, color='blue', s=12, label="Data points")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Best-fit line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# 7) Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='purple', s=12)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()
