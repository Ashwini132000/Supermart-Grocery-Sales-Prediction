# 🛒 Supermart Grocery Sales - Retail Analytics Dataset

This project aims to predict **sales amount** for grocery orders using **data analysis**, **feature engineering**, and **machine learning techniques**.
It leverages a fictional dataset — *Supermart Grocery Sales - Retail Analytics Dataset* — designed for practicing **retail analytics and ML modeling**.

---

## 📝 **Problem Statement**

The dataset, **Supermart Grocery Sales - Retail Analytics Dataset**, is a fictional dataset created for practicing **data analysis** and **machine learning**.
It contains **9,994 rows** of grocery order records placed by customers across various cities in **Tamil Nadu, India**.

| Column            | Description                                             |
| :---------------- | :------------------------------------------------------ |
| **Order ID**      | Unique identifier for each order                        |
| **Customer Name** | Name of the customer                                    |
| **Category**      | Product category (e.g., Food Grains, Beverages, Snacks) |
| **Sub Category**  | Detailed product type (e.g., Biscuits, Spices, Rice)    |
| **City**          | City where the order was placed                         |
| **Order Date**    | Date of the order                                       |
| **Region**        | Regional classification (North, South, East, West)      |
| **Sales**         | Total sales amount for the order                        |
| **Discount**      | Discount applied on the order                           |
| **Profit**        | Profit earned from the order                            |
| **State**         | State name (Tamil Nadu)                                 |

---

## **1️⃣ Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import timedelta, datetime

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
```

---

## **2️⃣ Data Loading and Preview**

```python
# Load the dataset
data = pd.read_csv('Supermart Grocery Sales - Retail Analytics Dataset.csv')
```
<img width="1122" height="591" alt="image" src="https://github.com/user-attachments/assets/8bdfa296-f9ea-496f-b3fb-526a2278c65a" />

- Dataset shape: **(9994, 11)**
- State column: Single value — *Tamil Nadu*

---

## **3️⃣ Data Preprocessing**

### 🔹 Convert and Format Dates

```python
data['Order_Date'] = pd.to_datetime(data['Order_Date'], format='mixed')
data['Order_Date'] = data['Order_Date'].dt.strftime('%d-%m-%Y')
```

Then re-convert and extract time-based components:

```python
data['Order_Date'] = pd.to_datetime(data['Order_Date'], format='%d-%m-%Y', errors='coerce')
data['Order_Day'] = data['Order_Date'].dt.day
data['Order_Month'] = data['Order_Date'].dt.month
data['Order_Year'] = data['Order_Date'].dt.year
data['Order_Weekday'] = data['Order_Date'].dt.dayofweek
data.drop('Order_Date', axis=1, inplace=True)
```
---

## 4️⃣ Exploratory Data Analysis (EDA)

To gain meaningful insights and understand key business patterns, **Exploratory Data Analysis (EDA)** was performed on the dataset.
The following visualizations helped uncover trends, relationships, and outliers affecting sales and profit.

### 📊 Key Analyses & Visuals

1. **Distribution of Sales by Category**

   * Compared total sales across major product categories.
   * Helped identify which category contributes the most to overall revenue.

2. **Top 10 Sub-Category Contribution**

   * Displayed top-performing sub-categories driving major sales.
   * Useful for identifying profitable product lines.

3. **Sales Distribution by Region**

   * Showed how sales vary across different geographical regions.
   * Revealed region-specific trends and high-performing zones.

4. **Top 10 Cities by Sales**

   * Highlighted cities with the highest total sales.
   * Helped identify key urban markets for strategic targeting.

5. **Monthly Sales Trend**

   * Analyzed monthly fluctuations to uncover **seasonal demand** patterns.
   * Crucial for planning inventory and marketing campaigns.

6. **Yearly Sales Growth**

   * Compared year-over-year performance.
   * Indicated business growth trajectory and market expansion.

7. **Profit vs Sales by Category**

   * Visualized profitability across categories.
   * Identified which categories generate high sales but low profits.

8. **Discount vs Profit by Category**

   * Explored the relationship between discount levels and profitability.
   * Helped in optimizing discount strategies to maintain profit margins.

*These plots provided actionable insights into product performance, customer segments, and pricing strategies.*

---

## 5️⃣ Outlier Detection & Treatment

Outliers can heavily distort the performance of regression models.
To ensure data quality, **outlier detection and capping** was performed on the numerical column `Profit` using the **IQR (Interquartile Range) method**.

<img width="1236" height="463" alt="image" src="https://github.com/user-attachments/assets/0342eea5-490b-412c-838b-31b8724ed5ef" />


---


## **6️⃣ Feature Engineering**

1. **Dropped Unnecessary Columns** – Columns such as `Order_ID`, `Customer_Name`, `State`, and `Order_Day` were removed.
2. **Created new features**  
   * Added **Quarter** indicator from `Order_Month`.
   * Calculated **Profit Margin** (`Profit / Sales`).
   * Calculated **Profit to Discount Ratio** (`Profit / Discount`).
3. **Defined Numeric and Categorical Features** for transformation.
4. **Used ColumnTransformer**  
   * Scaled numeric features using `StandardScaler`.
   * Encoded categorical features using `OneHotEncoder`.
5. **Split Data** into 80% training and 20% testing sets.

---

## 7️⃣ Model Implemented

| Model                           | Training Accuracy (R²) | Testing Accuracy (R²) |     MAE     |     MSE     |     RMSE    |
| :------------------------------ | :--------------------: | :-------------------: | :---------: | :---------: | :---------: |
| **Linear Regression**           |         0.5684         |         0.5649        |   296.0192  |  143506.91  |   378.8231  |
| **Decision Tree Regressor**     |         0.7828         |         0.7646        |   216.6352  |   77645.88  |   278.6501  |
| **Random Forest Regressor**     |         0.9903         |         0.9789        |   60.0983   |   6961.13   |   83.4334   |
| **XGBoost Regressor**           |      🏆 **0.9924**     |     🏆 **0.9879**     | **47.9639** | **4005.94** | **63.2925** |
| **Gradient Boosting Regressor** |         0.9856         |         0.9812        |   61.2544   |   6184.55   |   78.6419   |

---

### 📈 Insights

* The **XGBoost Regressor** achieved the **best overall performance**, with the highest R² score and the lowest error values.
* Ensemble models (**Random Forest**, **Gradient Boosting**, and **XGBoost**) significantly outperformed linear and decision tree models.
* Proper scaling and encoding using `ColumnTransformer` ensured consistent preprocessing across models.

---

## Feature Importance Insights — XGBoost Regressor

<img width="1312" height="842" alt="image" src="https://github.com/user-attachments/assets/046d3d99-5e4f-409f-b2a4-9f037cc2819f" />

### Key Findings

* **Profit_Discount_Ratio** is the most influential feature, indicating that sales strongly depend on how efficiently profits relate to discounts.
* **Profit_Margin** ranked second, showing that higher profitability contributes directly to better sales performance.
* **Discount** plays a crucial role — customers are highly sensitive to price reductions.
* **Sub_Category** features like *Cakes*, *Organic Fruits*, *Organic Vegetables*, and *Eggs* show moderate impact.
* **Regional and City-based** factors have minimal influence on overall sales.

---

## Business Interpretation

* **Profit-to-Discount Efficiency** drives sales more effectively than just providing high discounts.
* Maintaining a **healthy profit margin** while offering **strategic discounts** maximizes sales.
* Specific product groups (like *Cakes* and *Organic Products*) can be targeted for **focused marketing campaigns**.
* **Regional variations** exist but contribute less compared to financial indicators.

---

## Conclusion

This project successfully built a **Sales Prediction Model** using multiple machine learning algorithms.  
Key takeaways include:  

* **Feature Engineering** significantly improved model accuracy.
* **XGBoost** delivered superior predictive performance.
* **Profit efficiency and discount strategy** are the strongest drivers of sales outcomes.

These insights provide a **data-driven foundation for pricing strategy, inventory planning, and sales optimization**.

---

## 🧑‍💻 Author
**Ashwini Bawankar**  
*Data Science Enthusiast | Passionate about Machine Learning*

### 📬 Contact  
📧 Email: [abawankar13@gmail.com]  
🔗 LinkedIn: [https://www.linkedin.com/in/ashwini-bawankar/]
