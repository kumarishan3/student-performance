import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
#  Load data from CSV
# -----------------------------
df = pd.read_csv("data.csv")

print("📊 Basic Stats:\n", df.describe(), "\n")

# -----------------------------
#  Visualization
# -----------------------------
plt.figure()
scatter = plt.scatter(df['hours'], df['attendance'], c=df['pass'])
plt.xlabel("Study Hours")
plt.ylabel("Attendance")
plt.title("Student Performance")
plt.colorbar(scatter, label="Pass (0=Fail, 1=Pass)")
plt.show()

# -----------------------------
#  Features & Target
# -----------------------------
X = df[['hours', 'attendance']]
y = df['pass']

# -----------------------------
#  Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
#  Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, stratify=y, random_state=42
)

# -----------------------------
# Models
# -----------------------------
lr = LogisticRegression()
dt = DecisionTreeClassifier(random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)

# -----------------------------
#  Evaluation
# -----------------------------
print("📈 Logistic Accuracy:", accuracy_score(y_test, lr_pred))
print("📈 Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

print("\n🔍 Confusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, lr_pred))

print("\n📋 Classification Report:")
print(classification_report(y_test, lr_pred))

# -----------------------------
#  Custom Prediction
# -----------------------------
sample = scaler.transform([[5, 75]])
prediction = lr.predict(sample)[0]

print("\n🎯 Prediction for (5 hrs, 75% attendance):",
      "Pass" if prediction == 1 else "Fail")