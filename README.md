# 🚗 Car Price Predictor: Linear Regression with Real-World Data

This project predicts used car prices using a **carefully preprocessed real-world dataset** and a **Linear Regression model**. It includes everything from **data cleaning**, **feature engineering**, and **outlier removal** to **scaling**, **model training**, and **saving the final model pipeline** — all done in **pure Python with scikit-learn and pandas**.

---

## 🎯 Project Goal

To build a machine learning model that predicts the price of a car based on:
- Manufacturer (Make, Model, Trim)
- Mileage and Engine Size
- Fuel type, Gear type, Drivetrain, Condition
- Year of manufacture and more...

---

## 🚀 Major Highlights

### ✅ R² Score Boosted from 0.314 → 0.567
Through:
- Removing outliers using **IQR method**
- **Handling missing values**
- **One-hot encoding** categorical variables
- **Standardizing** features with `StandardScaler`

This model now explains over **56.7% of the variance** in car prices — a **massive improvement** for a baseline linear model working with messy real-world data.

---

### 🧠 Machine Learning Pipeline

| Step                        | Description |
|-----------------------------|-------------|
| 🧹 Data Cleaning            | Handled NaNs, dropped irrelevant features |
| 🧾 Feature Selection        | Retained only impactful predictors |
| 🔢 Categorical Encoding     | Used `pd.get_dummies()` with `drop_first=True` |
| 📉 Outlier Detection        | Removed extreme values using IQR |
| 📐 Feature Scaling          | Applied `StandardScaler` |
| 🧠 Model Training           | Trained `LinearRegression()` from scikit-learn |
| 📦 Model Saving             | Used `joblib` to save model, scaler & features |

---

## 📊 Model Performance

- **R² Score**: `0.567` on test set
- **Model type**: Linear Regression
- **Features used**: 13 (categorical + numerical)
- **Outlier Handling**: Interquartile Range (IQR)
- **Scaling**: StandardScaler

---

## 📁 Repository Structure
