import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

dashboard_df = pd.read_csv('bottle_new.csv', sep=',', low_memory=False)
dashboard_df.head(10)

dashboard_df.describe()

bottle_df = dashboard_df[['T_degC','Salnty']]

# And called again
bottle_df.columns = ['Temperature', 'Salinity']
bottle_df = bottle_df.dropna()
bottle_df = bottle_df[:][:500]

print(bottle_df.isnull().sum())

X = np.array(bottle_df['Salinity']).reshape(-1, 1)
y = np.array(bottle_df['Temperature']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

lin_df = LinearRegression()
lin_df.fit(X_train, y_train)

y_pred = lin_df.predict(X_test)                                     # Predict Linear Model
accuracy_score = lin_df.score(X_test, y_test)                       # Accuracy score
print("Linear Regression Model Accuracy Score: " + "{:.1%}".format(accuracy_score))

plt.scatter(X_test, y_test, color='r')
plt.plot(X_test, y_pred, color='g')
plt.show()
