import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
titanic = sns.load_dataset('titanic')

# Dataset info
print("=== Dataset Info ===")
titanic.info()  # info() prints automatically, no need for print()

# First 5 rows
print("\n=== First 5 Rows ===")
print(titanic.head())

# Last 5 rows
print("\n=== Last 5 Rows ===")
print(titanic.tail())

# Statistical summary
print("\n=== Statistical Summary ===")
print(titanic.describe())

# Transposed summary
print("\n=== Transposed Statistical Summary ===")
print(titanic.describe().transpose())

# Survived value counts
print("\n=== Survived Value Counts ===")
print(titanic['survived'].value_counts())

# Plot count of survived
print("\n=== Countplot of Survived ===")
sns.countplot(data=titanic, x='survived')
plt.show()

# Passenger class counts
print("\n=== Pclass Value Counts ===")
print(titanic['pclass'].value_counts())

# Passenger class by string label
print("\n=== Class Value Counts ===")
print(titanic['class'].value_counts())

# Combine pclass and class columns
p_class = titanic[['pclass', 'class']]
print("\n=== Pclass and Class Columns ===")
print(p_class.head())

# Value counts of 'sex'
print("=== Sex Value Counts ===")
print(titanic['sex'].value_counts())

# Countplot of 'sex'
print("\n=== Countplot of Sex ===")
sns.countplot(data=titanic, x='sex')
plt.show()

# Filter passengers younger than 20
less_than_20 = titanic[titanic['age'] < 20]

print("\n=== First 5 Passengers Age < 20 ===")
print(less_than_20.head())

print("\n=== Number of Passengers Age < 20 ===")
print(len(less_than_20))

# Pie chart of 'who'
print("\n=== Pie Chart of 'who' Column ===")
titanic.who.value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.show()

# Number of unique embark towns
print("\n=== Number of Unique 'embark_town' Values ===")
print(titanic['embark_town'].nunique())

# Check missing values
print("\n=== Missing Values in Dataset ===")
print(titanic.isnull().sum())

# Heatmap of missing values
print("\n=== Heatmap of Missing Values ===")
sns.heatmap(titanic.isnull(), yticklabels=False, cbar=False)
plt.show()

# Countplot of survived by sex
print("\n=== Survived Count by Sex ===")
sns.countplot(data=titanic, x='survived', palette='autumn', hue='sex')
plt.show()

# Countplot of survived by class
print("\n=== Survived Count by Class ===")
sns.countplot(data=titanic, x='survived', palette='viridis', hue='class')
plt.show()

# Histogram of age
print("\n=== Histogram of Age ===")
titanic['age'].plot(kind='hist', bins=30, color='green')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.show()


# Scatterplot: Age vs Fare colored by Class
print("=== Scatterplot: Age vs Fare by Class ===")
sns.scatterplot(data=titanic, x='age', y='fare', hue='class')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare by Class')
plt.show()

# Scatterplot: Age vs Fare colored by Sex
print("\n=== Scatterplot: Age vs Fare by Sex ===")
sns.scatterplot(data=titanic, x='age', y='fare', hue='sex')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare by Sex')
plt.show()

# Correlation matrix
print("\n=== Correlation Matrix ===")
correlation = titanic.corr(numeric_only=True)
print(correlation)

# Correlation of all features with 'survived'
print("\n=== Correlation with 'survived' ===")
print(correlation['survived'])

# Heatmap of correlation
print("\n=== Heatmap of Correlation Matrix ===")
plt.figure(figsize=(12,7))
sns.heatmap(correlation, annot=True, cmap='crest')
plt.title('Correlation Heatmap')
plt.show()