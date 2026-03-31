#import all modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.utils import resample
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report, r2_score,accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
import statsmodels.api as sm
from array import *
import plotly.express as px
import lightgbm as ltb
dataset = pd.read_csv(r'C:Downloads\New folder\supplement.csv')

#The block of code below is translating human concepts into pure math so the model can actually find patterns.
#I am basically deconstructing the 'Date' column to extract more data out of it like day,month,year,holiday,weekend,workingday etc.
dataset['Discount'] = dataset['Discount'].map({"No":0,"Yes":1})
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Year'] = dataset['Date'].dt.year
dataset['Month'] = dataset['Date'].dt.month
dataset['Day'] = dataset['Date'].dt.day
dataset['DayOfWeek'] = dataset['Date'].dt.dayofweek
dataset['DayOfYear'] = dataset['Date'].dt.dayofyear
dataset['Week'] = dataset['Date'].dt.isocalendar().week.astype(int)
dataset['Is_Weekend'] = np.where(dataset['DayOfWeek'] >= 5, 1, 0) 

#The concept used in the block of code below is called 'Feature Engineering'
# I am basically Combining multiple columns that are related to each other to craft meaningful data 
dataset['Discount_Holiday'] = dataset['Discount'] * dataset['Holiday']   # This basically creates a dataset which tells us the holidays on which discount was given on the orders
dataset['Store_Mean'] = dataset.groupby('Store_id')['#Order'].transform('mean') # Instead of letting the model make out the mean, I am doing it for it, this reduces time to run the model

#these are the features that i am going to 'feed' my model to train it
features = ['Store_id', 'Store_Type', 'Location_Type', 'Region_Code', 'Holiday', 'Discount', 'Year', 'Month', 'Day', 'DayOfWeek', 'Is_Weekend', 'DayOfYear', 'Week', 'Discount_Holiday', 'Store_Mean']

X = dataset[features].copy()
Y = dataset['#Order'].values

#Converting object/string columns to Pandas 'category' data types.
#This massively reduces RAM concumption as it converts the given data into a much more managable form of data for the model.
categorical_cols = ['Store_id', 'Store_Type', 'Location_Type', 'Region_Code', 'DayOfWeek', 'Month']
for col in categorical_cols:
    X[col] = X[col].astype('category')
    
#Train Test Split    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.18, random_state=30)
#model
model = ltb.LGBMRegressor(objective='regression', n_estimators=5000, learning_rate=0.015, num_leaves=255,max_depth=-1, subsample=0.8, colsample_bytree=0.8,random_state=40)
model.fit(X_train, Y_train, categorical_feature=categorical_cols)
y_pred = model.predict(X_test)
print("Final MAE:", mean_absolute_error(Y_test, y_pred))
print("Final R2 Score:", r2_score(Y_test, y_pred))           
# Top 10 most important features
ltb.plot_importance(model, max_num_features=10, importance_type='gain', figsize=(10, 6), color='teal')
plt.title("The Top 10 Drivers of Retail Sales")
plt.tight_layout()
plt.show()  
 
