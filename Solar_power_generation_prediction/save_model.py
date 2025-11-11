# import joblib
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_boston

# # Example: Train a RandomForestRegressor
# X, y = load_boston(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save the model
# joblib.dump(model, "random_forest_model.pkl")

# print("✅ Model saved as random_forest_model.pkl")



# save_model.py

from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd

# Load training data
df_train = pd.read_csv('Train_data.csv')
DC_cols = [col for col in df_train.columns if 'DC_POWER' in col]
AC_cols = [col for col in df_train.columns if 'AC_POWER' in col]

df_train['AC_POWER'] = df_train[AC_cols].sum(axis=1)/1000   # kW → MW
df_train['DC_POWER'] = df_train[DC_cols].sum(axis=1)/1000

x_train = df_train[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
y_train = df_train['AC_POWER']

# Train RandomForest model (or load the model if already trained)
rf_model = RandomForestRegressor(
    n_estimators=1100,
    min_samples_split=14,
    min_samples_leaf=8,
    max_features=None,
    max_depth=10,
    criterion='squared_error',
    random_state=42
)
rf_model.fit(x_train, y_train)

# Save the model
joblib.dump(rf_model, 'rf_model.pkl')
print("RandomForestRegressor model saved successfully!")

