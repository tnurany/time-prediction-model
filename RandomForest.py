import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print()
print(" Random Forest Regression:")
print(f" {'-'*len('Random Forest Regression:')}")
for n in [5, 10, 50, 100, 250, 500]:
	# Building and Training SVR Model
	regressor = RandomForestRegressor(n_estimators=n, random_state=1)
	regressor.fit(X_train, y_train)

	y_pred = regressor.predict(X_test)
	y_pred_train = regressor.predict(X_train)

	from sklearn.metrics import r2_score, mean_squared_error

	print(f" # of Trees = {n} R_Sq Score (Train Data) {r2_score(y_train, y_pred_train):.3f} ")
	print(f" # of Trees = {n} R_Sq Score (Test Data) {r2_score(y_test, y_pred):.3f} ")

	print(f" # of Trees = {n} MSE (Train Data) {mean_squared_error(y_train, y_pred_train):.2f} ")
	print(f" # of Trees = {n} MSE (Test Data) {mean_squared_error(y_test, y_pred):.2f} ")
	print()


