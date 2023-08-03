import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print()
print(" Polynomial Regression: ")
print(f" {'-'*len('Polynomial Regression:')}")

# Building and Training SVR Model
for i in [2, 3, 4]:
	poly_reg = PolynomialFeatures(degree=i)
	X_poly = poly_reg.fit_transform(X_train)
	regressor = LinearRegression()
	regressor.fit(X_poly, y_train)

	y_pred = regressor.predict(poly_reg.transform(X_test))
	y_pred_train = regressor.predict(X_poly)

	from sklearn.metrics import r2_score, mean_squared_error

	print(f" R_Sq Score for Degree {i} = {r2_score(y_test, y_pred):.3f}")
	print(f" MSE for Degree {i} =  {mean_squared_error(y_test, y_pred):.3f}")













