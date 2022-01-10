import pandas as pd
import numpy as np
import seaborn as sns
import math

########################
#Read in data and clean#
########################
argentina = pd.read_csv('data/argentina_cleaned.csv')
argentina.drop('Unnamed: 0', axis=1, inplace=True)
print("1/8 ... Data cleaned")

########################
#Create Baseline Models#
########################
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error

X_train, X_test, y_train, y_test = train_test_split(argentina.drop('price', axis=1), argentina.price, test_size=.3)

dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)
dummy_median = DummyRegressor(strategy = 'median').fit(X_train, y_train)

mean_preds = dummy_mean.predict(X_test)
median_preds = dummy_median.predict(X_test)

print("2/8 ... Baseline models created")

##############################
#Fix string columns, then ohe#
##############################
from sklearn.ensemble import GradientBoostingRegressor

#Convert necessary columns to datetime
argentina['end_date'] = pd.to_datetime(argentina['end_date'])
argentina['start_date'] = pd.to_datetime(argentina['start_date'])
argentina['created_on'] = pd.to_datetime(argentina['created_on'])

#OHE columns: l1, l2, l3, price_period, property_type, operation_type
arg2 = argentina.drop('l3', axis=1)

categorical_cols = ['l1', 'l2', 'property_type', 'operation_type'] 

arg3 = pd.get_dummies(arg2, columns = categorical_cols, dtype=int)

print("3/8 ... Fixed string columns, OHE done")

#####################
#Back to Modeling...#
#####################
X_train, X_test, y_train, y_test = train_test_split(arg3.drop(['price', 'start_date', 'end_date', 'created_on'], axis=1), arg3.price, test_size=.3)

gbc = GradientBoostingRegressor()
gbc.fit(X_train, y_train)
gbc_preds = gbc.predict(X_test)

gbr_rmse = mean_squared_error(y_test, gbc_preds)**(1/2)
gbr_mae = median_absolute_error(y_test, gbc_preds)
gbr_r2 = r2_score(y_test, gbc_preds)

print("4/8 ... Finished gradient boosting regressor")

#################################################
#Dummy Mean/Median Regressor on Transformed data#
#################################################
dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)
mean_preds = dummy_mean.predict(X_test)

dme_rmse = mean_squared_error(y_test, mean_preds)**(1/2)
dme_mae = median_absolute_error(y_test, mean_preds)
dme_r2 = r2_score(y_test, mean_preds)
#------------------------------------------------
dummy_median = DummyRegressor(strategy = 'median').fit(X_train, y_train)
median_preds = dummy_median.predict(X_test)

dmd_rmse = mean_squared_error(y_test, median_preds)**(1/2)
dmd_mae = median_absolute_error(y_test, median_preds)
dmd_r2 = r2_score(y_test, median_preds)

print("5/8 ... Finished dummy regressors on transformed data")

#########
#XGBoost#
#########
import xgboost as xgb

xgbr = xgb.XGBRegressor(n_estimators=100, learning_rate=.08, gamma=0, subsample=.75, colsample_bytree=1, max_depth=7)
xgbr.fit(X_train, y_train)
XGpreds = xgbr.predict(X_test)

xgb_rmse = mean_squared_error(y_test, XGpreds)**(1/2)
xgb_mae = median_absolute_error(y_test, XGpreds)
xgb_r2 = r2_score(y_test, XGpreds)

print("6/8 ... Finished XGBoost")

##################
#Lasso Regression#
##################
from sklearn import linear_model

las = linear_model.Lasso(max_iter=10000)
las.fit(X_train, y_train)
las_preds = las.predict(X_test)

las_rmse = mean_squared_error(y_test, las_preds)**(1/2)
las_mae = median_absolute_error(y_test, las_preds)
las_r2 = r2_score(y_test, las_preds)

print("7/8 ... Finished lasso regression")

###############
#Random Forest#
###############
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_preds = rfr.predict(X_test)

rfr_rmse = mean_squared_error(y_test, rfr_preds)**(1/2)
rfr_mae = median_absolute_error(y_test, rfr_preds)
rfr_r2 = r2_score(y_test, rfr_preds)

print("8/8 ... Finished random forest")

#################
#Models' Summary#
#################
data = [[dme_rmse, dme_mae, dme_r2], 
        [dmd_rmse, dmd_mae, dmd_r2],
        [gbr_rmse, gbr_mae, gbr_r2], 
        [xgb_rmse, xgb_mae, xgb_r2], 
        [las_rmse, las_mae, las_r2], 
        [rfr_rmse, rfr_mae, rfr_r2]]

Model_Performance = pd.DataFrame(data, index=['Dummy Mean Regressor','Dummy Median Regressor','Gradient Boosting Regressor','XGBoost','Lasso','Random Forest'],columns=['RMSE', 'MAE', 'R2'])

print(Model_Performance)






