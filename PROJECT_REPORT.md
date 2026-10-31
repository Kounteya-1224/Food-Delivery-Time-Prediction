# Project Report: Food Delivery Time Prediction

## Abstract
This project estimates food delivery duration from distance, number of food items, restaurant preparation time, traffic level, and weather. It uses multiple linear regression, gradient-based optimization, descriptive statistics, and held-out evaluation.

## Problem statement
Manual delivery-time estimates may not consistently account for order and travel conditions. The goal is to build a model that maps order features to expected delivery time in minutes.

## Objectives
1. Prepare a structured dataset for delivery orders.
2. Train a regression model using Gradient Descent.
3. Summarize numerical data using descriptive statistics and correlation.
4. Evaluate predictions with MAE, RMSE, and R².
5. check for possible overfitting by comparing training and testing errors.
6. Provide an interactive prediction interface.

## Methodology
1. Load the CSV data.
2. Split records into training and test subsets.
3. Standardize numeric features and one-hot encode categorical features.
4. Fit linear regression using squared-error loss and gradient descent.
5. Evaluate on data not used to fit the model.
6. Display prediction and diagnostic visualizations.

## Model
For input vector x and coefficient vector b:

ŷ = b0 + b1(distance) + b2(items) + b3(preparation time) + category effects

Mean squared error:

J(b) = (1/m) Σ(ŷᵢ − yᵢ)²

Gradient descent update:

bⱼ ← bⱼ − α ∂J/∂bⱼ

where α is the learning rate and m is the number of training examples.

## Statistics and metrics
- Mean: average value.
- Standard deviation: dispersion around the mean.
- Correlation: linear association between numeric variables.
- MAE: average absolute difference between prediction and actual time.
- RMSE: square root of mean squared error, more sensitive to large errors.
- R²: proportion of target variation explained relative to a mean-only baseline.

## Overfitting analysis
A much lower training error than testing error can indicate overfitting, although differences can also result from sampling noise or distribution shift. A single train/test split is only an initial check. Stronger evaluation uses cross-validation or a time-based test split, especially if future orders are the intended use.

## Results and limitation
The bundled CSV is synthetic demonstration data generated from a simple assumed relationship plus random noise. It makes the project runnable immediately, but scores on it are not evidence of real-world predictive performance. To assess practical accuracy, replace it with a sufficiently large, representative dataset of completed deliveries.

## Future improvements
- Use time-of-day, day-of-week, restaurant, and pickup delay.
- Use geographic route distance rather than straight-line distance.
- Use time-based validation and monitor performance over time.
- Compare linear regression with tree-based models.
- Quantify prediction uncertainty and communicate ranges rather than false precision.
