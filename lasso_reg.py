import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

lasso = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(alphas=np.logspace(-3, 1, 30), cv=5, max_iter=5000, random_state=42))
]).fit(Xtr, ytr)

yp = lasso.predict(Xte)
coef = lasso.named_steps["lasso"].coef_
mae = mean_absolute_error(yte, yp)
rmse = np.sqrt(mean_squared_error(yte, yp))
r2 = r2_score(yte, yp)

print("Best alpha:", lasso.named_steps["lasso"].alpha_)
print(f"Non-zero features: {(coef!=0).sum()}/{coef.size}")
print(f"MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

# Pred vs Actual
plt.scatter(yte, yp)
mn, mx = min(yte.min(), yp.min()), max(yte.max(), yp.max())
plt.plot([mn, mx], [mn, mx], linestyle='--')
plt.title("Lasso: Predicted vs Actual"); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.show()

# Coefficients (sparse)
plt.bar(range(len(coef)), coef)
plt.title("Lasso Coefficients (many zeros possible)"); plt.xlabel("Feature index"); plt.ylabel("Weight"); plt.show()
