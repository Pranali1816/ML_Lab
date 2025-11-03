# ===========================
# Diabetes Prediction Project
# ===========================

# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===========================
# Step 1: Load the dataset
# ===========================
df1 = pd.read_csv("diabetes.csv")   # Ensure diabetes.csv is in the same directory
print("\nFirst 5 rows of dataset:")
print(df1.head())

# ===========================
# Step 2: Data Information
# ===========================
print("\nDataset Info:")
print(df1.info())

print("\nStatistical Summary:")
print(df1.describe())

# ===========================
# Step 3: Check for missing values
# ===========================
plt.figure(figsize=(8,5))
sns.heatmap(df1.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# ===========================
# Step 4: Data Visualization
# ===========================
sns.set_style('whitegrid')

# Count of diabetic (1) vs non-diabetic (0)
plt.figure(figsize=(6,4))
sns.countplot(x='Outcome', data=df1, palette='cubehelix')
plt.title("Count of Diabetic vs Non-Diabetic Patients")
plt.show()

# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(df1['Age'], bins=30, color='darkblue')
plt.title("Age Distribution")
plt.show()

# Blood pressure distribution
plt.figure(figsize=(6,4))
sns.histplot(df1['BloodPressure'], bins=20, color='royalblue')
plt.title("Blood Pressure Distribution")
plt.show()

# Relationship between Age and BloodPressure
sns.jointplot(x='Age', y='BloodPressure', data=df1, kind='scatter', color='green')
plt.show()

# Barplot: Average BloodPressure by Age (aggregated)
plt.figure(figsize=(15,8))
sns.barplot(x="Age", y="BloodPressure", data=df1, errorbar=None, palette="crest")
plt.title("Blood Pressure by Age")
plt.show()

# ===========================
# Step 5: Data Splitting
# ===========================
X = df1.drop('Outcome', axis=1)
y = df1['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=101
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# ===========================
# Step 6: Model Training
# ===========================
LRModel = LogisticRegression(solver='lbfgs', max_iter=7600)
LRModel.fit(X_train, y_train)
print("\nModel training completed successfully!")

# ===========================
# Step 7: Model Evaluation
# ===========================
predictions_diabetes = LRModel.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, predictions_diabetes))

print("Accuracy Score:", round(accuracy_score(y_test, predictions_diabetes)*100, 2), "%")

# Confusion Matrix Visualization
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, predictions_diabetes), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ===========================
# Step 8: Predict New Patient Data
# ===========================
new_data = {
    'Pregnancies': [0],
    'Glucose': [170],
    'BloodPressure': [126],
    'SkinThickness': [60],
    'Insulin': [35],
    'BMI': [30.1],
    'DiabetesPedigreeFunction': [0.649],
    'Age': [78]
}

patient_data = pd.DataFrame(new_data)
print("\nNew Patient Data:")
print(patient_data)

prediction = LRModel.predict(patient_data)
print("\nPrediction for the new patient (1 = Diabetic, 0 = Non-Diabetic):",prediction[0])
