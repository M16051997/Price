#!/usr/bin/env python
# coding: utf-8

# #                                             # PREDICTING THE  PRICES
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Load Required dataset

# In[2]:


data= pd.read_csv(r"C:\Users\M.komala\Downloads\Expenses - Sheet1.csv")


# # Perform EDA

# In[3]:


data.head()


# In[4]:


data.region.value_counts()


# In[5]:


data.sex.value_counts()


# In[6]:


data.info()


# In[7]:


data.shape


# In[8]:


data.isnull().values.any()


# In[9]:


data.describe()


# In[10]:


data.skew().sort_values(ascending=False)


# In[11]:


import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()


# # Plot for charges on a histogram. 
# 
# The distribution of sale prices is right skewed, something that is expected.
# Here I perform my first bit of feature engineering. I’ll apply a log transform to charges to compress outliers making the distribution normal.
# 
# Outliers can have devastating effects on models that use loss functions minimising squared error. Instead of removing outliers try applying a transformation.

# In[12]:


from scipy import stats
# transform training data & save lambda value
charges_log, fitted_lambda = stats.boxcox(data['charges'])
charges_log


# In[13]:


# creating axes to draw plots
fig, ax = plt.subplots(1, 2)
 
# plotting the original data(non-normal) and
# fitted data (normal)
sns.distplot(data['charges'], hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2},
            label = "Non-Normal", color ="green", ax = ax[0])
 
sns.distplot(charges_log, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2},
            label = "Normal", color ="green", ax = ax[1])
 
# adding legends to the subplots
plt.legend(loc = "upper right")
 
# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)
 
print(f"Lambda value used for Transformation: {fitted_lambda}")


# In[14]:


data['charges_log']= np.log(data.charges)


# In[15]:


x = data.charges
sns.set_style('whitegrid')
sns.distplot(x)
plt.show()

data['charges_log'] = np.log(data.charges)
x = data.charges_log
sns.distplot(x)
plt.show()


# In[16]:


data= data.drop(columns=['charges'])


# In[17]:


data.head()


# # Correlation of Data

# In[18]:


# mask out upper triangle
mask = np.zeros_like(data.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# heatmap
sns.heatmap(data.corr()*100, 
           cmap='RdBu_r', 
           annot = True, 
           mask = mask)


# In[19]:


print(data.corr())


# # We have some Categorical Data so apply One hot Encoding

# In[20]:


data=pd.get_dummies(data)
data.head(2)


# In[21]:


plt.scatter(data['bmi'], data['smoker_no'], color='red')


# In[22]:


plt.scatter(data['bmi'], data['smoker_yes'], color='red')


# # Superwised Machine Learning Models

# # Build Linear Regression Model

# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Data Cross Validation by using Train test split Method
# Cross-validation is a resampling method that uses different portions of the data to test and train a model on different iterations.

# In[24]:


# Split X and y

X = data.drop(['charges_log'], axis=1)
y = data[['charges_log']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[25]:


X.head(2)


# In[26]:


print("Train Data", X_train.shape)
print("Test Data", X_test.shape)


# In[27]:


model= LinearRegression(n_jobs=-1)
model.fit(X_train, y_train)
y_pred= model.predict(X_test)


# Explore the Predicted Values in graphical format

# In[28]:


plt.scatter(y_test[0:10], y_pred[0:10], color='red')


# In[29]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

explained_variance=metrics.explained_variance_score(y_test, y_pred)
mean_absolute_error=metrics.mean_absolute_error(y_test, y_pred) 
mse=metrics.mean_squared_error(y_test, y_pred) 
mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
median_absolute_error=metrics.median_absolute_error(y_test, y_pred)
r2_linear=metrics.r2_score(y_test, y_pred)

print('explained_variance: ', round(explained_variance,4))    
print('mean_squared_log_error: ', round(mean_squared_log_error,4))
print('r2: ', round(r2_linear,4))
print('MAE: ', round(mean_absolute_error,4))
print('MSE: ', round(mse,4))
print('RMSE: ', round(np.sqrt(mse),4))


# # Linear Regression Model Accuracy

# In[30]:


print('Linear Regression Model Accuracy is', r2_linear.round(2)*100, '%')


# In[ ]:





# # Build Ridge Regression Model

# In[31]:


from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha=0.001, normalize=True)
ridgeReg.fit(X_train,y_train)
print(sqrt(mean_squared_error(y_train, ridgeReg.predict(X_train))))
print(sqrt(mean_squared_error(y_test, ridgeReg.predict(X_test))))
r2_ridge=ridgeReg.score(X_test, y_test)
print('R2 Value/Coefficient of Determination: {}'.format(ridgeReg.score(X_test, y_test)))


# # Ridge Regression Model Accuracy

# In[32]:


print('Ridge Regression Model Accuracy is', r2_ridge.round(2)*100, '%')


# In[ ]:





# # Build Lasso Regression Model

# In[33]:


from sklearn.linear_model import Lasso
lassoreg = Lasso(alpha=0.001, normalize=True)
lassoreg.fit(X_train,y_train)
r2_lasso=lassoreg.score(X_test, y_test)

print(sqrt(mean_squared_error(y_train, lassoreg.predict(X_train))))
print(sqrt(mean_squared_error(y_test, lassoreg.predict(X_test))))
print('R2 Value/Coefficient of Determination: {}'.format(lassoreg.score(X_test, y_test)))


# # Lasso Regression Model Accuracy

# In[34]:


print('Lasso Regression Model Accuracy is', r2_lasso.round(2)*100, '%')


# In[ ]:





# # Build Elastic Net Regression Model

# In[35]:


from sklearn.linear_model import ElasticNet
Elas = ElasticNet(alpha=0.001, normalize=True)
Elas.fit(X_train, y_train)
r2_Elastic=Elas.score(X_test, y_test)
print(sqrt(mean_squared_error(y_train, Elas.predict(X_train))))
print(sqrt(mean_squared_error(y_test, Elas.predict(X_test))))
print('R2 Value/Coefficient of Determination: {}'.format(Elas.score(X_test, y_test)))


# # Elastic Net Regression Model Accuracy

# In[36]:


print('Elastic Net Regression Model Accuracy is', r2_Elastic.round(2)*100, '%')


# In[ ]:





# # Build XGB Regressor

# # The benefit of using ensembles of decision tree methods like gradient boosting is that they can automatically provide estimates of feature importance from a trained predictive model

# In[37]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from numpy import absolute


# In[38]:


import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse


# In[39]:


# Instantiate an XGBoost object with hyperparameters
xgb_reg = xgb.XGBRegressor(max_depth=3, n_estimators=100, n_jobs=2,
                           objectvie='reg:squarederror', booster='gbtree',
                           random_state=42, learning_rate=0.05)

# Train the model with train data sets
xgb_reg.fit(X_train, y_train)

y_pred = xgb_reg.predict(X_test) # Predictions
y_true = y_test # True values

MSE = mse(y_true, y_pred)
RMSE = np.sqrt(MSE)


# In[40]:


# trained XGBoost model automatically calculates feature importance on our predictive modeling problem.

# These importance scores are available in the feature_importances_ member variable of the trained model.


# # Model importance 
# Importance provides a score that indicates how useful or valuable each feature was in the construction of the boosted decision trees within the model. The more an attribute is used to make key decisions with decision trees, the higher its relative importance.
# 
# This importance is calculated explicitly for each attribute in the dataset, allowing attributes to be ranked and compared to each other.
# 
# Importance is calculated for a single decision tree by the amount that each attribute split point improves the performance measure, weighted by the number of observations the node is responsible for. The performance measure may be the purity (Gini index) used to select the split points or another more specific error function.
# 
# The feature importances are then averaged across all of the the decision trees within the model.

# In[41]:


print(xgb_reg.feature_importances_)


# In[42]:


R_squared = r2_score(y_true, y_pred)

print("\nRMSE: ", np.round(RMSE, 2))
print()
print("R-Squared: ", np.round(R_squared, 2)*100, '%')


# # XGB Regressor Model Accuracy

# In[43]:


print('XGB Regression Model Accuracy is', R_squared.round(2)*100, '%')


# In[ ]:





# # Decision Tree Model

# In[44]:


# import the regressor
from sklearn.tree import DecisionTreeRegressor

# create a regressor object
regressor = DecisionTreeRegressor(random_state = 0)

# fit the regressor with X and Y data
regressor.fit(X_train, y_train)


# In[45]:


# predicting value
predictions = regressor.predict(X_test)


# In[46]:


r2_deci = r2_score(y_test, predictions)


# In[47]:


r2_deci


# In[48]:


plt.scatter(y_test[0:15], predictions[0:15], color='red')


# # Decision Tree Model Accuracy

# In[49]:


print('Decision Tree Regression Model Accuracy is', r2_deci.round(2)*100, '%')


# In[ ]:





# # Random Forest Model

# In[50]:


# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor

# create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

# fit the regressor with x and y data
regressor.fit(X_train, y_train)


# In[51]:


Predictionsrandomforest = regressor.predict(X_test) # test the output by changing values


# In[52]:


plt.scatter(y_test[0:15], Predictionsrandomforest[0:15], color='red')


# In[53]:


r2 = r2_score(y_test,Predictionsrandomforest)
r2


# In[54]:


print('Random Forest Regression Model Accuracy is ', r2.round(2)*100, '%')


# In[ ]:





# # Conclusion
# 
# By Comparing the all Models XGB Regressor is the best fith 84% Accuracy

# In[ ]:





# In[ ]:




