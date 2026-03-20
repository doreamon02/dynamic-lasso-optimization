import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# STEP 1: LOAD DATA
# =========================
df = pd.read_csv("train.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# =========================
# STEP 2: CLEAN DATA
# =========================
if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

# Target
y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)

# Handle missing values
X = X.fillna(X.median(numeric_only=True))
X = X.fillna("None")

# Convert categorical to numeric
X = pd.get_dummies(X, drop_first=True)

print("After encoding:", X.shape)

# =========================
# STEP 3: TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# STEP 4: SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# STEP 5: BASELINE MODELS
# =========================
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

lasso = Lasso(alpha=0.01, max_iter=5000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

# =========================
# STEP 6: CUSTOM MODEL
# =========================
class DynamicLasso:
    def __init__(self, alpha=0.01, lambda0=0.1, gamma=0.01, max_iter=1000):
        self.alpha = alpha
        self.lambda0 = lambda0
        self.gamma = gamma
        self.max_iter = max_iter

    def soft_threshold(self, z, tau):
        return np.sign(z) * np.maximum(np.abs(z) - tau, 0)

    def fit(self, X, y):
        n, p = X.shape
        beta = np.zeros(p)

        for k in range(self.max_iter):
            lam = self.lambda0 / (1 + self.gamma * k)
            grad = (1/n) * X.T @ (X @ beta - y)
            z = beta - self.alpha * grad
            beta = self.soft_threshold(z, self.alpha * lam)

        self.coef_ = beta

    def predict(self, X):
        return X @ self.coef_

# Train custom model
model = DynamicLasso()
model.fit(X_train, y_train)
dynamic_pred = model.predict(X_test)

# =========================
# STEP 7: EVALUATION
# =========================
def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return [name, mse, rmse, r2]

results = []
results.append(evaluate("Ridge", y_test, ridge_pred))
results.append(evaluate("LASSO", y_test, lasso_pred))
results.append(evaluate("Dynamic LASSO", y_test, dynamic_pred))

results_df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "R2 Score"])

print("\n=== MODEL COMPARISON ===")
print(results_df)

# =========================
# STEP 8: PLOTS
# =========================
plt.figure()
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("Model Comparison (RMSE)")
plt.show()

plt.figure()
plt.scatter(y_test, dynamic_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (Dynamic LASSO)")
plt.show()
plt.figure()
plt.plot(model.coef_)
plt.title("Model Coefficients (Feature Weights)")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.show()
ridge_features = np.sum(np.abs(ridge.coef_) > 1e-6)
lasso_features = np.sum(np.abs(lasso.coef_) > 1e-6)
dynamic_features = np.sum(np.abs(model.coef_) > 1e-6)

names = ["Ridge", "LASSO", "Dynamic"]
values = [ridge_features, lasso_features, dynamic_features]

plt.figure()
plt.bar(names, values)
plt.title("Selected Features Comparison")
plt.ylabel("Number of Features")
plt.show()