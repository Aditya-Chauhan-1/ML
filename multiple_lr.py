import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.DataFrame({
    "Size": [1400,1600,1700,1875,1100,1500,2400,3000,1850,1200],
    "Bedrooms": [3,3,3,4,2,3,4,5,4,2],
    "Age": [10,5,7,3,20,12,2,1,4,25],
    "Price": [245000,312000,279000,308000,199000,265000,450000,570000,330000,210000]
})
X = data[["Size","Bedrooms","Age"]]; y = data["Price"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
m = LinearRegression()
m.fit(Xtr, ytr)
yp = m.predict(Xte)
print(yte)
print(yp)

mae = mean_absolute_error(yte, yp)
rmse = np.sqrt(mean_squared_error(yte, yp))
r2 = r2_score(yte, yp)

print("Coefficients [Size, Bedrooms, Age]:", m.coef_)
print("Intercept:", m.intercept_)
print(f"MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

# Pred vs Actual
plt.scatter(yte, yp)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], linestyle='--')
plt.title("Predicted vs Actual"); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.show()

# Residuals
res = yte - yp
plt.scatter(yp, res)
plt.axhline(0, linestyle='--')
plt.title("Residuals vs Predicted"); plt.xlabel("Predicted"); plt.ylabel("Residuals"); plt.show()
