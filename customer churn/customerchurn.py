#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

# read the dataset
churn_df = pd.read_csv("churn.csv")
churn_df.head()


# In[ ]:


# check if there is null value
churn_df.info()


# In[ ]:


# descritive statistics
churn_df.describe()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# visualization
fig = plt.figure()
fig.set(alpha = 0.3)
plt.subplot2grid((1,2),(0,0))
churn_df["Churn?"].value_counts().plot(kind = "bar")
# title and others...
plt.title("Is churn?")
plt.ylabel("number")

plt.subplot2grid((1,2),(0,1))
churn_df['CustServ Calls'].value_counts().plot(kind = "bar")
# title and others...
plt.title("CustoServ Calls")
plt.ylabel("number")


# In[ ]:


plt.subplot2grid((1,3),(0,0))
churn_df['Day Mins'].plot(kind = 'kde')
# title and others...
plt.title("Density of Day mins")
plt.xlabel("Day Mins")
plt.ylabel("Density")

plt.subplot2grid((1,3),(0,1))
churn_df['Day Calls'].plot(kind = 'kde')
# title and others...
plt.title("Density of Day calls")
plt.xlabel("Day Calls")
plt.ylabel("Density")

plt.subplot2grid((1,3),(0,2))
churn_df['Day Charge'].plot(kind = 'kde')
# title and others...
plt.title("Density of Day charge")
plt.xlabel("Day Charge")
plt.ylabel("Density")

plt.show()


# In[ ]:


# visualization with int'l plan
int_true = churn_df["Churn?"][churn_df["Int\'l Plan"] == "yes"].value_counts()
int_false = churn_df["Churn?"][churn_df["Int\'l Plan"] == "no"].value_counts()

df_int = pd.DataFrame({"int plan": int_true, "no int plan": int_false})

df_int.plot(kind = "bar", stacked = True)
plt.title("Is churm?")
plt.ylabel("number")

plt.show()


# In[ ]:


# visualization with custserv calls
cus_stay = churn_df["CustServ Calls"][churn_df["Churn?"] == "False."].value_counts()
cus_churn = churn_df["CustServ Calls"][churn_df["Churn?"] == "True."].value_counts()

df_cus = pd.DataFrame({"stay": cus_stay, "churn": cus_churn})
df_cus.plot(kind = "bar", stacked = True)
plt.title("Visualization of churn and custserv calls")
plt.ylabel("number")
plt.xlabel("service call")


# In[ ]:


# data transformation
churn_df["Churn?"] = np.where(churn_df["Churn?"] == "False.", 0, 1)
churn_df = pd.get_dummies(churn_df, columns = ["Int'l Plan"], prefix = ["int'l plan"])
churn_df = pd.get_dummies(churn_df, columns = ["VMail Plan"], prefix = ["vmail plan"])

# data cleanning
churn_df = churn_df.drop(["State", "Area Code", "Phone", "Account Length"], axis = 1)


# In[ ]:


churn_df.head()


# In[ ]:


# adjust the orders of columns
cols = list(churn_df)
cols.insert(0,cols.pop(cols.index("Churn?")))
churn_df= churn_df.loc[:,cols]
churn_df.head()


# In[ ]:


churn_df.dtypes


# In[ ]:


# normalisation
features = churn_df.iloc[:,1:].columns
X = churn_df.iloc[:,1:].values.astype(np.float)
Y = churn_df["Churn?"]

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

print("Observations: %d, Features : %d." %X.shape)
#print("Unique target labels: ", np.unique(Y))
print("Customer churn: " + str(Y.sum()) + " stay: " + str(len(Y)-Y.sum()))


# In[ ]:


# Split the X and Y
from sklearn.model_selection import KFold
def run_cv(X, Y, clf_class, **kwargs):
    kf = KFold(n_splits = 5, shuffle = True)
    Y_pred = Y.copy()
        
    #conduct the validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train = Y[train_index]
         #type of classification
        clf = clf_class(**kwargs)
         #training
        clf.fit(X_train, Y_train)
        #predict
        Y_pred[test_index] = clf.predict(X_test)
    return Y_pred


# In[ ]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

#calculate the aacuracy of the result of prediction
def accuracy(Y_true, Y_pred):
    return np.mean(Y_true==Y_pred)

print("Accuracy of SVM: %.2f" %accuracy(Y, run_cv(X, Y, SVC)))
print("Accuracy of RandomForest: %.2f" %accuracy(Y, run_cv(X, Y, RF)))
print("Accuracy of KNeighbors: %.2f" %accuracy(Y, run_cv(X, Y, KNN)))


# In[ ]:


def run_prob_cv(X, Y, clf_class, **kwargs):
    kf = KFold(n_splits = 5, shuffle = True)
    Y_prob = np.zeros((len(Y), 2))
    for train_index, test_index in kf.split(Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train = Y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, Y_train)
        Y_prob[test_index] = clf.predict_proba(X_test)
    return Y_prob


# In[ ]:


#calculate the recall
pred_prob = run_prob_cv(X, Y, RF, n_estimators = 10)
pred_churn = pred_prob[:, 1]
is_churn = Y==1

counts =pd.value_counts(pred_churn)

true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    true_prob = pd.Series(true_prob)
    
counts = pd.concat([counts, true_prob], axis = 1).reset_index()
counts.columns = ["pred_prob", "count", "true_prob"]
counts

