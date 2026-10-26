# Sonar Rock vs Mine Prediction Model

This repository contains a **Machine Learning workflow using Logistic Regression** to classify sonar returns from underwater objects into two categories: **Rocks (R)** or **Mines (M)**. 

## 📊 Dataset Overview
The model uses the standard Sonar dataset (`sonar.csv`), which contains data collected by bouncing sonar signals off various surfaces at different angles.
- **Total Samples:** 208
- **Features:** 60 numerical attributes (representing frequencies and sonar bounce configurations)
- **Target Variable (Column 60):** Binary labels:
  - `R`: Rock
  - `M`: Mine

## 🛠️ Model Architecture
- **Algorithm:** Logistic Regression
- **Libraries Used:** NumPy, Pandas, Scikit-Learn
- **Data Splitting:** 90% Training / 10% Testing (Stratified by label distribution)

## 📈 Performance
- **Training Accuracy:** ~83.42%
- **Test Accuracy:** ~76.19%

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python installed along with the required scientific libraries:
```bash
pip install numpy pandas scikit-learn
```

### 2. Implementation Script
You can use the following script to load data, train the model, and make a new inference pipeline:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load data
sonar_data = pd.read_csv('sonar.csv', header=None)

# 2. Split features and labels
X = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1, stratify=y
)

# 4. Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Evaluate Performance
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print(f"Training Accuracy: {accuracy_score(train_preds, y_train):.2%}")
print(f"Test Accuracy: {accuracy_score(test_preds, y_test):.2%}")

# 6. Making a Prediction on Custom Data
input_data = (
    0.0453, 0.0523, 0.0843, 0.0689, 0.1183, 0.2583, 0.2156, 0.3481, 0.3337, 
    0.2872, 0.4918, 0.6552, 0.6919, 0.7797, 0.7464, 0.9444, 1.0000, 0.8874, 
    0.8024, 0.7818, 0.5212, 0.4052, 0.3957, 0.3914, 0.3250, 0.3200, 0.3271, 
    0.2767, 0.4423, 0.2028, 0.3788, 0.2947, 0.1984, 0.2341, 0.1306, 0.4182, 
    0.3835, 0.1057, 0.1840, 0.1970, 0.1674, 0.0583, 0.1401, 0.1628, 0.0621, 
    0.0203, 0.0530, 0.0742, 0.0409, 0.0061, 0.0125, 0.0084, 0.0089, 0.0048, 
    0.0094, 0.0191, 0.0140, 0.0049, 0.0052, 0.0044
)

# Reshape array for a single instance prediction
input_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_array)

if prediction[0] == 'R':
    print("The object is identified as a Rock.")
else:
    print("The object is identified as a Mine.")
```
