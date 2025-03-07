import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Task 4/advertising.csv")

print("Dataset Preview:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

new_data = pd.DataFrame([[230, 37, 69]], columns=['TV', 'Radio', 'Newspaper'])
predicted_sales = model.predict(new_data)
print(f"\nPredicted Sales for TV=230, Radio=37, Newspaper=69: {predicted_sales[0]:.2f}")

plt.figure(figsize=(8,5))
index = np.arange(len(y_test))
plt.bar(index, y_test, width=0.4, label="Actual Sales", color='blue', alpha=0.7)
plt.bar(index + 0.4, y_pred, width=0.4, label="Predicted Sales", color='red', alpha=0.7)
plt.xlabel("Index")
plt.ylabel("Sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.show()
