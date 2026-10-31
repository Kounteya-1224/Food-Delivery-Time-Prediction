# Food Delivery Time Prediction

A complete educational ML project with a Streamlit UI, Linear Regression optimized using Gradient Descent, statistics, evaluation, and overfitting checks.

## Important data note
`data/food_delivery_sample.csv` is **synthetic demo data** generated for this project. It is not a real delivery-company dataset. Performance metrics from it are illustrative and should not be presented as real-world accuracy.

## Features
- Inputs: distance, number of items, restaurant preparation time, traffic, weather
- Output: predicted delivery time in minutes
- Numeric feature scaling and categorical one-hot encoding
- Gradient Descent using scikit-learn `SGDRegressor` with squared-error loss
- Mean, standard deviation, correlations
- MAE, RMSE, R², actual-vs-predicted plot, residual plot
- Training-vs-testing error comparison for an overfitting check
- CSV dataset download and input prediction interface

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser, usually at `http://localhost:8501`.

## Replace with real data
Replace `data/food_delivery_sample.csv` with a CSV containing these exact columns:
- `distance_km`
- `num_items`
- `prep_time_min`
- `traffic_level` (Low/Medium/High)
- `weather` (Clear/Rainy/Stormy)
- `delivery_time_min` (actual completed delivery time)

Use enough representative records, check missing values and outliers, and evaluate on a later-time test set when possible. For real delivery deployment, account for time of day, restaurant, rider availability, pickup delays, and local geography where appropriate.

## Project structure
```
food_delivery_time_prediction/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── food_delivery_sample.csv
```
