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
plt.show()```
✅ Findings: The data follows a near-normal distribution with a slight right skew.

### 3️⃣ Model Building & Training
📌 Model 1: Multiple Linear Regression
Used cnt as the dependent variable and various features as independent variables
Model Evaluation Metrics:
Mean Squared Error (MSE): 556,776
R² Score: 0.833
