import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5))
]).fit(Xtr, ytr)

yp = ridge.predict(Xte)
mae = mean_absolute_error(yte, yp)
rmse = np.sqrt(mean_squared_error(yte, yp))
r2 = r2_score(yte, yp)
alpha = ridge.named_steps["ridge"].alpha_
coef = ridge.named_steps["ridge"].coef_

print("Best alpha:", alpha)
print(f"MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

# Pred vs Actual
plt.scatter(yte, yp)
mn, mx = min(yte.min(), yp.min()), max(yte.max(), yp.max())
plt.plot([mn, mx], [mn, mx], linestyle='--')
plt.title("Ridge: Predicted vs Actual"); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.show()

# Coefficients
plt.bar(range(len(coef)), coef)
plt.title("Ridge Coefficients"); plt.xlabel("Feature index"); plt.ylabel("Weight"); plt.show()
