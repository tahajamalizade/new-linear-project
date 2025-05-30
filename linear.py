import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

url = "https://github.com/dataprofessor/data/raw/master/BostonHousing.csv"
df = pd.read_csv(url)

X = df.drop(columns="medv")
Y = df["medv"]                

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("✅ Coefficients:", model.coef_)
print("✅ Intercept:", model.intercept_)
print("✅ Mean Squared Error (MSE): %.2f" % mean_squared_error(Y_test, Y_pred))
print("✅ R² Score: %.2f" % r2_score(Y_test, Y_pred))

plt.figure(figsize=(8,6))
sns.scatterplot(x=Y_test, y=Y_pred, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Linear Regression)")
plt.grid(True)
plt.show()
