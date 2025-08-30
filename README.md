Great choice 🚢—the Titanic survival prediction is one of the most popular beginner machine learning projects. Let’s walk step by step through building a classification model to predict survival.

We’ll use Python, Pandas, Scikit-Learn, and Matplotlib/Seaborn for data analysis and modeling.


---

📝 Step-by-Step Titanic Survival Prediction

1. Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


---

2. Load Dataset

Download Titanic dataset (train.csv) from Kaggle Titanic Dataset.

# Load dataset
data = pd.read_csv("train.csv")

# Preview
print(data.head())
print(data.info())


---

3. Data Preprocessing

Drop unnecessary columns like PassengerId, Name, Ticket, Cabin.

Handle missing values (Age, Embarked).

Encode categorical variables (Sex, Embarked).


# Drop unused columns
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Fill missing Age with median
data["Age"].fillna(data["Age"].median(), inplace=True)

# Fill missing Embarked with most frequent value
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])
data["Embarked"] = le.fit_transform(data["Embarked"])

print(data.head())


---

4. Split Features & Target

# Features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


---

5. Train Model (Logistic Regression)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


---

6. Evaluate Model

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


---

7. Example Visualization

# Survival count by gender
sns.countplot(x="Survived", hue="Sex", data=data, palette="Set2")
plt.title("Survival Distribution by Gender")
plt.show()

# Survival by passenger class
sns.barplot(x="Pclass", y="Survived", data=data, palette="muted")
plt.title("Survival Rate by Passenger Class")
plt.show()


---

✅ End Result:
You’ll have a model that predicts whether a passenger survived the Titanic disaster based on their features. Logistic Regression usually gives ~78–82% accuracy.


---

Would you like me to also show you how to improve this model with Random Forest / Decision Trees (Task 2 style extension), or do you want to keep Task 1 limited to Logistic Regression only?

