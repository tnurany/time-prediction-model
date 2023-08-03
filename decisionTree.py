import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor

# Import the dataset
df = pd.read_csv("./data/production_line_data_finalV2.csv")

df = df.drop(["STOCKNO"], axis=1)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encoding
# Using Label Encoder
vehicle_le = LabelEncoder()
X[:, 1] = vehicle_le.fit_transform(X[:, 1])

# Using One Hot Encoding
station_ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), [2])],
                               remainder='passthrough')
X = np.array(station_ct.fit_transform(X))



# Splitting Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Building and Training SVR Model
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

np.set_printoptions(precision=2)
# print(np.concatenate([y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)], axis=1))

from sklearn.metrics import r2_score,mean_squared_error
print()
print(" Decision Tree Regression: ")
print(f" {'-'*len('Decision Tree Regression:')}")
print(f" R_Sq Score (Train Data): {r2_score(y_train, y_pred_train):.3f}")
print(f" R_Sq Score (Test Data): {r2_score(y_test, y_pred):.3f}")
print(f" MSE (Train Data): {mean_squared_error(y_train, y_pred_train):.2f}")
print(f" MSE (Test Data): {mean_squared_error(y_test, y_pred):.2f}")

print()
print(" Depth of the Tree: ", regressor.get_depth())