# 🚴‍♂️ Bike Sharing Demand Prediction Project

## 📌 Project Overview  

This project aims to **predict bike rental demand** using machine learning models.  
By leveraging the **Bike Sharing Dataset**, we apply various regression models (Linear Regression, LASSO, Logistic Regression) and explore the best-performing model.

- **Dataset Used**: [Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset)  
- **Technologies**: `Python`, `scikit-learn`, `statsmodels`, `pandas`, `matplotlib`, `seaborn`  
- **Objective**: Predict **bike rental counts (`cnt`)** based on weather, date, and other factors  

---

## 🛠 Project Workflow  

### 1️⃣ Data Preprocessing
- Converted `dteday` (date) and categorical features (`season`, `weathersit`) into **numerical values**
- Applied **scaling** to `temp` (temperature), `hum` (humidity), and `windspeed`
- Used **One-Hot Encoding** to convert categorical variables like `season` and `weathersit`

### 2️⃣ Exploratory Data Analysis (EDA)
- Analyzed the distribution of bike rental counts (`cnt`)
- Identified correlations between features and bike rentals
- **Visualized key insights**  

#### 📊 Data Distribution Visualization  
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing the distribution of bike rentals
plt.figure(figsize=(8, 6))
sns.histplot(day['cnt'], kde=True, bins=30)
plt.title("Bike Rental Count Distribution")
plt.xlabel("Bike Rentals (cnt)")
plt.ylabel("Frequency")
plt.show()
```

✅ Findings: The data follows a near-normal distribution with a slight right skew.

### 3️⃣ Model Building & Training  

#### 📌 Model 1: Multiple Linear Regression  
Used `cnt` as the **dependent variable** and various features as **independent variables**.  

📊 **Model Evaluation Metrics:**  
- **Mean Squared Error (MSE):** `556,776`  
- **R² Score:** `0.833`  

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Model training
model = LinearRegression()
input_features = ['yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
model.fit(day_train[input_features], day_train['cnt'])

# Predictions on test data
predictions = model.predict(day_test[input_features])

# Model evaluation
mse = mean_squared_error(day_test['cnt'], predictions)
r2 = r2_score(day_test['cnt'], predictions)

print("MSE:", mse)
print("R^2:", r2)
```
📊 **Actual vs. Predicted Bike Rentals:**  

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(day_test['cnt'], predictions, alpha=0.5)
plt.xlabel("Actual Bike Rentals")
plt.ylabel("Predicted Bike Rentals")
plt.title("Actual vs. Predicted Bike Rentals")
plt.show()
```
#### 📌 Model 2: LASSO Regression (L1 Regularization)  
- Applied LASSO regression to eliminate irrelevant variables
- Optimal λ (alpha) value = 50.0
- Selected features: summer, winter, temp, hum, windspeed, days_since_2011
  
📊 **Model Performance:**  
- **Mean Squared Error (MSE):** `644,382`  
- **R² Score:** `0.831`
  
```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=50.0)
lasso.fit(day_train[input_features], day_train['cnt'])

# Evaluation
test_predictions = lasso.predict(day_test[input_features])
mse = mean_squared_error(day_test['cnt'], test_predictions)
r2 = r2_score(day_test['cnt'], test_predictions)

print("MSE:", mse)
print("R^2:", r2)
```
📊 **LASSO Regression Coefficient Changes**  
```python
import numpy as np

lambdas = np.logspace(-2, 4, 100)
coefs = []

for l in lambdas:
    lasso = Lasso(alpha=l)
    lasso.fit(day_train[input_features], day_train['cnt'])
    coefs.append(lasso.coef_)

plt.figure(figsize=(10, 6))
plt.plot(np.log10(lambdas), coefs)
plt.xlabel("log10(λ)")
plt.ylabel("Regression Coefficients")
plt.title("LASSO Regression Coefficient Changes")
plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
plt.grid(True)
plt.show()

```
✅ Findings: LASSO regression removes unnecessary variables, keeping only key predictors like temp, windspeed, and hum.

### 📌 Conclusion  

✔ **LASSO Regression is the final model of choice**  
✔ Achieves an **R² score of 0.8354** while removing non-essential features  

#### 🔑 **Key Predictors Influencing Bike Rentals**  
- **Temperature (`temp`)**: Higher temperatures **increase** bike rentals  
- **Humidity (`hum`) & Wind Speed (`windspeed`)**: Higher values **reduce** bike rentals  
- **Days since 2011 (`days_since_2011`)**: Captures **long-term trends** in demand  
- **Rental counts drop significantly in summer & winter** → **Marketing opportunities**  

### ✅ **Business Insights**  

✔ **Bike rental demand is highly seasonal** → Consider **seasonal promotions** 📅  
✔ **Weather has a major impact** → Integrate **weather forecasts** into service 🌦  
✔ **Weekdays have lower demand than weekends** → Potential for **weekday promotions** 📉  


