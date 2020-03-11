#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate,train_test_split,KFold,LeaveOneOut,cross_val_score,cross_val_predict
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge 
from sklearn.preprocessing import StandardScaler 
import seaborn as sns
from scipy import stats


# In[2]:


url = 'https://raw.githubusercontent.com/orangeczw/ML_team_Name/master/Task_1a/train.csv'
train = pd.read_csv(url, error_bad_lines=False)
y = train.iloc[:,1].copy()
X = train.iloc[:,2:].copy()
#See if there's NA value
X.isna().any()


# In[12]:


#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
cor[abs(cor)>0.8]#Find out highly correlated independent variables  


# In[13]:


#Correlation with output variable
cor_target = abs(cor["y"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.3]
relevant_features


# In[14]:


X_del = X.drop(['x9'], axis=1)
#delete x9 since cor(x9,x10) = 0.910228, and x9 is less correlated with y than x10.


# In[15]:


# scaling the inputs 
scaler = StandardScaler() 
scaled_X = scaler.fit_transform(X_del) 

penalty = [0.01,0.1,1,10,100]
RMSE = []

for lmd in penalty:
    model = Ridge(alpha = lmd, normalize = False, tol = 0.001, solver ='auto', random_state = 42) #solver?random_state?
    cv_results = cross_validate(model, scaled_X, y, cv=10,
                            n_jobs=-1,scoring=('neg_mean_squared_error'))
    result = np.sqrt(np.abs(cv_results["test_score"]).mean())
    RMSE.append(result)
    
RMSE = pd.DataFrame(RMSE) 
# RMSE.to_csv(r'C:\Users\l\Desktop\task_1a_2.csv', index = False, header=False)
np.savetxt("submission.csv",np.array(RMSE))
print(RMSE)


# In[ ]:




