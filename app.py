"""
Food Delivery Time Prediction
Educational project using Linear Regression trained with Gradient Descent.
The bundled dataset is SYNTHETIC demo data, not real delivery-company data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "food_delivery_sample.csv"

st.set_page_config(page_title="Food Delivery Time Predictor", page_icon="🍱", layout="wide")
st.title("🍱 Food Delivery Time Prediction")
st.caption("Linear Regression • Gradient Descent • Statistics • Overfitting Analysis")
st.info("Demo note: the included dataset is synthetic. Predictions are for learning/demo purposes and are not real-world guarantees.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model(data, test_size, learning_rate, epochs):
    features = ["distance_km", "num_items", "prep_time_min", "traffic_level", "weather"]
    target = "delivery_time_min"
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    numeric = ["distance_km", "num_items", "prep_time_min"]
    categorical = ["traffic_level", "weather"]
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])
    # Explicit gradient-descent implementation via SGDRegressor.
    from sklearn.linear_model import SGDRegressor
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", SGDRegressor(
            loss="squared_error",
            penalty=None,
            learning_rate="constant",
            eta0=learning_rate,
            max_iter=epochs,
            tol=1e-5,
            random_state=42,
            early_stopping=False
        ))
    ])
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    metrics = {
        "train_mae": mean_absolute_error(y_train, pred_train),
        "test_mae": mean_absolute_error(y_test, pred_test),
        "train_rmse": np.sqrt(mean_squared_error(y_train, pred_train)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, pred_test)),
        "test_r2": r2_score(y_test, pred_test)
    }
    return model, X_train, X_test, y_train, y_test, pred_train, pred_test, metrics

data = load_data()
with st.sidebar:
    st.header("Training settings")
    test_size = st.slider("Test set fraction", 0.10, 0.40, 0.20, 0.05)
    learning_rate = st.select_slider("Gradient Descent learning rate", options=[0.0005, 0.001, 0.005, 0.01, 0.03], value=0.01)
    epochs = st.slider("Training iterations (epochs)", 100, 3000, 1000, 100)
    st.caption("Lower learning rates may converge more slowly; very high rates may be unstable.")

model, X_train, X_test, y_train, y_test, pred_train, pred_test, metrics = train_model(
    data, test_size, learning_rate, epochs
)

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Dataset", "Statistics", "Model evaluation"])

with tab1:
    st.subheader("Enter order details")
    c1, c2, c3 = st.columns(3)
    with c1:
        dist = st.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)
    with c2:
        item_count = st.number_input("Number of items", min_value=1, max_value=100, value=3, step=1)
    with c3:
        prep_time = st.number_input("Restaurant preparation time (min)", min_value=1, max_value=180, value=15, step=1)
    c4, c5 = st.columns(2)
    with c4:
        traffic = st.selectbox("Traffic level", ["Low", "Medium", "High"])
    with c5:
        weather = st.selectbox("Weather condition", ["Clear", "Rainy", "Stormy"])
    if st.button("Predict delivery time", type="primary"):
        order = pd.DataFrame([{
            "distance_km": dist, "num_items": item_count, "prep_time_min": prep_time,
            "traffic_level": traffic, "weather": weather
        }])
        estimate = float(model.predict(order)[0])
        st.success(f"Estimated delivery time: **{max(1, estimate):.1f} minutes**")
        st.caption("Estimate comes from the currently trained demo model. Real accuracy requires representative real delivery data.")

with tab2:
    st.subheader("Dataset preview")
    st.write(f"Rows: **{len(data)}** · Columns: **{len(data.columns)}**")
    st.dataframe(data.head(25), use_container_width=True)
    st.download_button("Download dataset CSV", data.to_csv(index=False), "food_delivery_data.csv", "text/csv")
    st.caption("Replace this sample CSV with real, cleaned historical orders before using predictions operationally.")

with tab3:
    st.subheader("Descriptive statistics")
    st.dataframe(data[["distance_km", "num_items", "prep_time_min", "delivery_time_min"]].describe().round(2), use_container_width=True)
    st.subheader("Correlation with delivery time")
    numeric_corr = data[["distance_km", "num_items", "prep_time_min", "delivery_time_min"]].corr(numeric_only=True)
    fig, ax = plt.subplots()
    im = ax.imshow(numeric_corr, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_corr.columns)), numeric_corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(numeric_corr.index)), numeric_corr.index)
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Correlation indicates association, not causation. Encoded categories are evaluated by the model but omitted from this numeric-only matrix.")

with tab4:
    st.subheader("Evaluation on held-out test data")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test MAE", f"{metrics['test_mae']:.2f} min")
    m2.metric("Test RMSE", f"{metrics['test_rmse']:.2f} min")
    m3.metric("Test R²", f"{metrics['test_r2']:.3f}")
    st.subheader("Training vs testing error (overfitting check)")
    e1, e2 = st.columns(2)
    e1.metric("Training MAE", f"{metrics['train_mae']:.2f} min")
    e2.metric("Testing MAE", f"{metrics['test_mae']:.2f} min")
    gap = metrics["test_mae"] - metrics["train_mae"]
    if gap > max(3.0, metrics["train_mae"] * 0.5):
        st.warning("The testing error is notably higher than training error. Investigate overfitting, data leakage, or train/test distribution differences.")
    else:
        st.info("The train/test MAE gap is not large by the selected heuristic. This alone does not prove the model generalizes well.")
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, pred_test, alpha=0.65)
    lo = min(float(y_test.min()), float(np.min(pred_test)))
    hi = max(float(y_test.max()), float(np.max(pred_test)))
    ax2.plot([lo, hi], [lo, hi], linestyle="--")
    ax2.set_xlabel("Actual delivery time (min)")
    ax2.set_ylabel("Predicted delivery time (min)")
    ax2.set_title("Actual vs predicted")
    fig2.tight_layout()
    st.pyplot(fig2)
    residuals = y_test.to_numpy() - pred_test
    fig3, ax3 = plt.subplots()
    ax3.hist(residuals, bins=25)
    ax3.set_xlabel("Residual (actual − predicted), minutes")
    ax3.set_ylabel("Count")
    ax3.set_title("Residual distribution")
    fig3.tight_layout()
    st.pyplot(fig3)

with st.expander("Model details"):
    st.markdown("""
    - **Algorithm:** Linear Regression optimized with stochastic Gradient Descent (`SGDRegressor`, squared-error loss).
    - **Preprocessing:** StandardScaler for numerical features; OneHotEncoder for traffic and weather.
    - **Statistics:** Descriptive statistics and numerical-feature correlations.
    - **Evaluation:** MAE, RMSE, R², actual-vs-predicted plot, residual histogram.
    - **Overfitting analysis:** Compare training and held-out testing MAE; use more validation and real data for stronger conclusions.
    """)
