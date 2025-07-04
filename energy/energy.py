import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("12-house_energy_regression.csv")
print(df.head())

X=df.drop("avg_indoor_temp_change", axis=1)
y=df["avg_indoor_temp_change"]
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,train_size=0.25,random_state=15)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
from sklearn.neighbors import KNeighborsRegressor
regressor=KNeighborsRegressor(n_neighbors=5,algorithm="auto")
regressor.fit(X_train_scaled,y_train)
y_pred=regressor.predict(X_test_scaled)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("avg_indoor_temp_change r2_score ",r2_score(y_test,y_pred))
print("avg_indoor_temp_change mean_absolute_erro",mean_absolute_error(y_test,y_pred))
print("avg_indoor_temp_change mean_squared_error",mean_squared_error(y_test,y_pred))
plt.scatter(y_test,y_pred)
plt.show()

X=df.drop("outdoor_humidity_level", axis=1)
y=df["outdoor_humidity_level"]
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,train_size=0.25,random_state=15)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
from sklearn.neighbors import KNeighborsRegressor
regressor=KNeighborsRegressor(n_neighbors=5,algorithm="auto")
regressor.fit(X_train_scaled,y_train)
y_pred=regressor.predict(X_test_scaled)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("outdoor_humidity_level r2_score ",r2_score(y_test,y_pred))
print("outdoor_humidity_level mean_absolute_erro",mean_absolute_error(y_test,y_pred))
print("outdoor_humidity_level mean_squared_error",mean_squared_error(y_test,y_pred))

plt.scatter(y_test,y_pred)
plt.show()
X=df.drop("daily_energy_consumption_kwh", axis=1)
y=df["daily_energy_consumption_kwh"]
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=15)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
from sklearn.neighbors import KNeighborsRegressor
regressor=KNeighborsRegressor(n_neighbors=5,algorithm="auto")
regressor.fit(X_train_scaled,y_train)
y_pred=regressor.predict(X_test_scaled)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("daily_energy_consumption_kwh r2_score ",r2_score(y_test,y_pred))
print("daily_energy_consumption_kwh mean_absolute_erro",mean_absolute_error(y_test,y_pred))
print("daily_energy_consumption_kwh mean_squared_error",mean_squared_error(y_test,y_pred))

plt.scatter(y_test,y_pred)
plt.show()

df_a=pd.read_csv("12-health_risk_classification.csv")
print(df_a.head())
X=df_a.drop("high_risk_flag",axis=1)
y=df_a["high_risk_flag"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15)

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,algorithm="auto",weights="uniform")
classifier.fit(X_train_scaled, y_train)
y_pred=classifier.predict(X_test_scaled)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(" high_risk_flag confusion_matrix",confusion_matrix(y_test,y_pred))
print(" high_risk_flag accuracy_score",accuracy_score(y_test,y_pred))
print(" high_risk_flag classification_report",classification_report(y_test,y_pred))
