import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


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


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding
# Using Label Encoder on the Vehicle Variable
vehicle_le = LabelEncoder()
X[:, 1] = vehicle_le.fit_transform(X[:, 1])

# Using One Hot Encoding on Station Variable
station_ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), [2])],
                               remainder='passthrough')
X = np.array(station_ct.fit_transform(X))


# Splitting Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)

y_pred = regressor.predict(poly_reg.transform(X_test))

predicted_data = pd.DataFrame(np.concatenate([y_test.reshape(len(y_test), 1),
                                              y_pred.reshape(len(y_pred), 1)
                                              ],
                                             axis=1))

predicted_data.rename(columns={0: "Original Time",
                               1: "Predicted Time",},
                      inplace=True)

predicted_data.to_csv("./prediction/predicted_data.csv", index=False)

print("Write Complete")



