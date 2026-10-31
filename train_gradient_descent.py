"""Train/evaluate a linear regression model with batch Gradient Descent from scratch."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "food_delivery_sample.csv")
X = df[["distance_km", "num_items", "prep_time_min", "traffic_level", "weather"]]
y = df["delivery_time_min"].to_numpy(dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pre = ColumnTransformer([
    ("num", StandardScaler(), ["distance_km", "num_items", "prep_time_min"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["traffic_level", "weather"])
])
X_train = pre.fit_transform(X_train)
X_test = pre.transform(X_test)
# Convert sparse one-hot output to dense for this small educational dataset.
if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
    X_test = X_test.toarray()

# Add intercept column.
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Batch Gradient Descent. Feature scaling helps the optimizer converge.
weights = np.zeros(X_train.shape[1])
learning_rate = 0.03
epochs = 3000
m = len(y_train)
loss_history = []

for epoch in range(epochs):
    predictions = X_train @ weights
    errors = predictions - y_train
    gradient = (2 / m) * (X_train.T @ errors)
    weights -= learning_rate * gradient
    if epoch % 50 == 0:
        loss_history.append(float(np.mean((X_train @ weights - y_train) ** 2)))

train_pred = X_train @ weights
test_pred = X_test @ weights

print("Batch Gradient Descent Linear Regression")
print(f"Training MAE: {mean_absolute_error(y_train, train_pred):.2f} min")
print(f"Testing MAE : {mean_absolute_error(y_test, test_pred):.2f} min")
print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.2f} min")
print(f"Testing R^2 : {r2_score(y_test, test_pred):.3f}")
print("Final training MSE:", round(float(np.mean((train_pred-y_train)**2)), 3))
print("Note: bundled dataset is synthetic demo data, not real-world evidence.")
