import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR

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
station_ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), [2])], remainder='passthrough')
X = np.array(station_ct.fit_transform(X))


# Splitting Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Building and Training SVR Model
svr_regressor_linear = SVR(kernel='linear')
svr_regressor_linear.fit(X_train, y_train)

svr_regressor_poly = SVR(kernel='poly')
svr_regressor_poly.fit(X_train, y_train)

svr_regressor_rbf = SVR(kernel='rbf')
svr_regressor_rbf.fit(X_train, y_train)

svr_regressor_sigmoid = SVR(kernel='sigmoid')
svr_regressor_sigmoid.fit(X_train, y_train)


y_pred_linear = svr_regressor_linear.predict(X_test)
y_pred_linear_train = svr_regressor_linear.predict(X_train)

y_pred_poly = svr_regressor_poly.predict(X_test)
y_pred_poly_train = svr_regressor_poly.predict(X_train)

y_pred_rbf = svr_regressor_rbf.predict(X_test)
y_pred_rbf_train = svr_regressor_rbf.predict(X_train)

y_pred_sigmoid = svr_regressor_sigmoid.predict(X_test)
y_pred_sigmoid_train = svr_regressor_sigmoid.predict(X_train)

# print(np.concatenate([y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)], axis=1))

from sklearn.metrics import r2_score

print(" R_2 Scores from Support Vector Regression")
print(f" {'-'*len('R_2 Scores from Support Vector Regression')}")
print(" Linear Kernel (Train Data): ", r2_score(y_train, y_pred_linear_train))
print(" Linear Kernel (Test Data)", r2_score(y_test, y_pred_linear))

print(" Poly Kernel (Train Data): ", r2_score(y_train, y_pred_poly_train))
print(" Poly Kernel (Test Data): ", r2_score(y_test, y_pred_poly))

print(" RBF Kernel (Train Data): ", r2_score(y_train, y_pred_rbf_train))
print(" RBF Kernel (Test Data):", r2_score(y_test, y_pred_rbf))

print(" Sigmoid Kernel (Train Data): ", r2_score(y_train, y_pred_sigmoid_train))
print(" Sigmoid Kernel (Test Data):", r2_score(y_test, y_pred_sigmoid))

