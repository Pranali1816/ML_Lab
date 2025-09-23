# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Employee.csv')
print("Initial Dataset Shape:", df.shape)
print("Initial Dataset Head:")
print(df.head())

# Check and remove duplicate rows
print("Number of duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Dataset after removing duplicates Shape:", df.shape)
print(df.head())

# Check dataset info
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values in Dataset:")
print(df.isna().sum())

# Check unique values
print("\nUnique Values per Column:")
unique = df.nunique()
print(unique)

# Clean inconsistent entries in categorical data
df['Company'] = df['Company'].replace('Infosys Pvt Lmt', 'Infosys')
df['Company'] = df['Company'].replace('Tata Consultancy Services', 'TCS')
df['Company'] = df['Company'].replace('CTS', 'Congnizant')
df['Company'] = df['Company'].replace('Congnizant', 'Cognizant')
df['Place'] = df['Place'].replace('Podicherry', 'Pondicherry')

print("\nDataset after cleaning categorical inconsistencies:")
print(df[['Company', 'Place']].head())

# Outlier detection using IQR for 'Age'
q1_age = df['Age'].quantile(0.25)
q3_age = df['Age'].quantile(0.75)
iqr_age = q3_age - q1_age
lower_age = q1_age - 1.5 * iqr_age
upper_age = q3_age + 1.5 * iqr_age
outliers_age = df[(df['Age'] < lower_age) | (df['Age'] > upper_age)]
print("\nOutliers in Age column:")
print(outliers_age)

# Outlier detection using IQR for 'Salary'
q1_sal = df['Salary'].quantile(0.25)
q3_sal = df['Salary'].quantile(0.75)
iqr_sal = q3_sal - q1_sal
lower_sal = q1_sal - 1.5 * iqr_sal
upper_sal = q3_sal + 1.5 * iqr_sal
outliers_salary = df[(df['Salary'] < lower_sal) | (df['Salary'] > upper_sal)]
print("\nOutliers in Salary column:")
print(outliers_salary)

# Handle missing values
df['Company'] = df['Company'].fillna(df['Company'].mode()[0])
df['Age'] = df['Age'].replace(0, np.nan)
rounded_mean_age = round(df['Age'].mean(), 0)
df['Age'] = df['Age'].fillna(rounded_mean_age)
rounded_mean_salary = round(df['Salary'].mean(), 0)
df['Salary'] = df['Salary'].fillna(rounded_mean_salary)
df['Place'] = df['Place'].fillna(df['Place'].mode()[0])

print("\nDataset after handling missing values:")
print(df.head())

# Filter specific condition
filtered_df = df[(df['Age'] > 40) & (df['Salary'] < 5000)]
print("\nEmployees with Age > 40 and Salary < 5000:")
print(filtered_df)

# Scatter plot: Age vs Salary
plt.scatter(df['Age'], df['Salary'], color='g')
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary")
plt.show()

# Bar plot: Count of employees by city
x = df['Place'].value_counts().index
y = df['Place'].value_counts().values
plt.bar(x, y, color='m')
plt.xlabel("Cities")
plt.ylabel("No: of Employees")
plt.title("Employees from Each City")
plt.xticks(rotation=45)
plt.show()

# Gender Encoding
df['Gender'] = df['Gender'].replace(0, 'M')
df['Gender'] = df['Gender'].replace(1, 'F')
print("\nDataset after Gender Encoding:")
print(df[['Gender']].head())

# Drop irrelevant column
df = df.drop('Country', axis=1)
print("\nDataset after dropping 'Country':")
print(df.head())

# OneHotEncoding for Company, Place, and Gender
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
df_array = ohe.fit_transform(df[['Company', 'Place', 'Gender']]).toarray()

# Get categories and flatten
categories = ohe.categories_
single_list = [item for sublist in categories for item in sublist]

# Create new DataFrame with encoded columns
df_new = pd.DataFrame(df_array, columns=single_list)
print("\nOneHotEncoded Data Head:")
print(df_new.head())

# Concatenate original and encoded data
df_ml = pd.concat([df, df_new], axis=1)
print("\nFinal Dataset after concatenating OneHotEncoded columns:")
print(df_ml.head())

# Save cleaned dataset
df.to_csv('Employee_cleaned.csv', index=False)
print("\nCleaned dataset saved as 'Employee_cleaned.csv'")

# Label Encoding for 'Company'
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df_sample = pd.read_csv('Employee_cleaned.csv')
df_sample['Company'] = label.fit_transform(df_sample['Company'])
print("\nDataset after Label Encoding 'Company':")
print(df_sample[['Company']].head())

# Standardization using StandardScaler
from sklearn.preprocessing import StandardScaler
data = df[['Age', 'Salary']]
scaler = StandardScaler().fit(data)
data_scaled = scaler.transform(data)
scaled_data_set = pd.DataFrame(data_scaled, columns=data.columns)
print("\nStandardized Data Head:")
print(scaled_data_set.head())

# Min-Max Normalization using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler_mm = MinMaxScaler().fit(data)
data_scaled_mm = scaler_mm.transform(data)
mm_scaled_data_set = pd.DataFrame(data_scaled_mm, columns=data.columns)
print("\nMin-Max Scaled Data Head:")
print(mm_scaled_data_set.head())
