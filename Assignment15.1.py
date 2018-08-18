
# coding: utf-8

# In[68]:


#Program to build a linear regression model using sklearn model for boston data
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
#Importing all the libraries required for building a linear regression model
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Load the boston data
boston = load_boston()
bos = pd.DataFrame(boston.data)
#Get the column names and price from the target column.
bos.columns = boston.feature_names
bos['PRICE'] = boston.target
#Take Y axis as the price column and other as the X axis.
X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']
#Split the dataset into the test/train data set.
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size = 0.278, random_state = 5)
#LinearRegression model creation.
lm = LinearRegression()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)
plt.scatter(Y_test, Y_pred,color='blue')
plt.xlabel("Prices: $Y_i$",fontweight="bold",color='red')
plt.ylabel("Predicted Prices: ${Yp}_i$",fontweight="bold",color='red')
plt.title("Prices vs Predicted prices: $Y_i$ vs ${Yp}_i$",fontweight="bold",color='blue')
plt.show()
#To check the level of error of a model using mean squared error
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)

