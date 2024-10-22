import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

df = pd.read_csv('synthetic_house_prices.csv')

print(df.head())

print(df.isnull().sum())

X = df.drop('Price', axis=1)  
y = df['Price']               

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

selector = RFE(LinearRegression(), n_features_to_select=5)  
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

lin_model = LinearRegression()
lin_model.fit(X_train_selected, y_train)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_selected, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_selected, y_train)

models = {'Linear Regression': lin_model,
          'Decision Tree': dt_model,
          'Random Forest': rf_model}

results = {}

for name, model in models.items():
    y_pred = model.predict(X_test_selected)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy_percentage = r2 * 100 
    results[name] = {'MSE': mse, 'R²': r2, 'Accuracy (%)': accuracy_percentage}
    print(f'{name} - Mean Squared Error: {mse:.2f}, R-squared Score: {r2:.4f}, Accuracy: {accuracy_percentage:.2f}%')

plt.scatter(y_test, rf_model.predict(X_test_selected))
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices (Random Forest)')
plt.show()

print("\nSummary of Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: MSE = {metrics['MSE']:.2f}, R² = {metrics['R²']:.4f}, Accuracy = {metrics['Accuracy (%)']:.2f}%")