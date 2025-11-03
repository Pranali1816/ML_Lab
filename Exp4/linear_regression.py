import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load data
HouseDF = pd.read_csv('USA_Housing.csv')

# Drop non-numeric column
HouseDF = HouseDF.drop('Address', axis=1)

# Basic exploration
print(HouseDF.info())
print(HouseDF.describe())

# Pairplot
sns.pairplot(HouseDF)
plt.show()

# Heatmap
sns.heatmap(HouseDF.corr(), annot=True, cmap='coolwarm')
plt.show()

# Features and target
X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
             'Avg. Area Number of Bedrooms', 'Area Population']]
y = HouseDF['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Train model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Coefficients
print("Intercept:", lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# Predictions
predictions = lm.predict(X_test)

# Visualization
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

sns.histplot((y_test - predictions), bins=50, kde=True)
plt.show()

# Evaluation
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
