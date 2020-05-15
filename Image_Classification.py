#IMPORTING DEPENDECIES
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline
#using pandas to read the database stored in another folder.
data = pd.read_csv(r"E:\python\self\__pycache__\mnist_test.csv")
#the r is written to convert normal string to raw string
#viewing column heads 
data.head()
#Extracting data from datset and viewing them close 
a= data.iloc[3,1:].values
a = a.reshape(28,28)
plt.imshow(a)
#preparing the data
#separating labels and values
df_x = data.iloc[:,1:] #Except first column print all
df_y =data.iloc[:,0]
#creating train and test size of model
x_train,x_test,y_train,y_test = train_test_split(df_x , df_y, test_size = 0.2, random_state = 4)
#Check data
y_train.head()
#call rf classifier 
rf = RandomForestClassifier(n_estimators = 100)
#now check how well our classifier has worked 
#prediction on test data
pred = rf.predict(x_test)
pred
#check prediction accuracy
a = y_test.values
#calculate total no of correct predicted values
count = 0
for i in range(len(pred)):
    if pred[i]== a[i]:
        count = count+1
count
len(pred)
#accuracy value
1901/2000
#We have 95.05 % accuracy with MNIST dataset using RandomForestClassifier
        
