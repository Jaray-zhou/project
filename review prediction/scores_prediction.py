#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

reviews_scores_df = pd.read_csv("reviews_scores_prediction.csv")
reviews_scores_df.head()


# In[2]:


# drop the unrelated column
reviews_scores_df = reviews_scores_df.drop(["Unnamed: 0"], axis = 1)
reviews_scores_df.head()


# In[3]:


# Separate the target values(Y) from predictors(X)
X = reviews_scores_df.iloc[:, reviews_scores_df.columns != "review_score"]
Y = reviews_scores_df["review_score"]

# create the list of scores for testing data
scores_test = {}

# split data into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
		test_size = 0.2, random_state=5)  # X is “1:” and Y is “[0]”

# print the shapes to check everything is OK
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#mba_df.loc[:, mba_df.columns != 'b']


# In[4]:


# check whether the data is an inbalnced sample or not
Y_train.value_counts()


# In[5]:


from collections import Counter
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
# Oversampling
print('Original dataset shape %s' % Counter(Y))
sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(X, Y)
print('Resampled dataset shape %s' % Counter(Y_res))

# split the X and Y again
X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, 
		test_size = 0.2, random_state=5)  # X is “1:” and Y is “[0]”

# print the shapes to check everything is OK
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
# check whether the data is an inbalnced sample or not
Y_train.value_counts()


# In[6]:


from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
get_ipython().run_line_magic('matplotlib', 'inline')

np.set_printoptions(precision=2)

# creat congusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalise=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          multi=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalise=True`.
    """
    if not title:
        if normalise:
            title = 'Normalised confusion matrix'
        else:
            title = 'Confusion matrix, without normalisation'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    if multi==True:
    	classes = classes[unique_labels(y_true, y_pred)]
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor");

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax


# In[7]:


from sklearn.tree import DecisionTreeClassifier as DTC

# a decision tree model with default values
dtc = DTC()
# fit the model using some training data
dtc_fit = dtc.fit(X_train, Y_train)
# generate a mean accuracy score for the predicted data
train_score = dtc.score(X_train, Y_train)
# print the mean accuracy of testing predictions
print("DTC mean accuracy (Train) = " + str(round(train_score, 4)))


# In[8]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier as DTC

# tune the hyperparameters for DTC
tuned_parameters = [{'criterion': ['gini', 'entropy'],
                     'max_depth': [3, 5, 7],
                     'min_samples_split': [3, 5, 7],
                     'max_features': ["sqrt", "log2", None]}]

scores = ['accuracy', 'f1_macro', 'recall_macro']

for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(DTC(), tuned_parameters, cv=5,
                       scoring= score)
    clf.fit(X_train, Y_train)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")


# In[69]:


# check the accuracy metrics again
# a decision tree model with default values
dtc = DTC('gini', max_depth= 7, max_features= None, min_samples_split= 5)

# fit the model using some training data
dtc_fit = dtc.fit(X_train, Y_train)

# generate a mean accuracy score for the predicted data
train_score = dtc.score(X_train, Y_train)
# print the mean accuracy of training dataset
print("DTC mean accuracy (after tuning hyperparameters)(Train) = " + str(round(train_score, 4)))

# predict the test data
predicted = dtc_fit.predict(X_test)

# generate a mean accuracy score for the predicted data
test_score = dtc_fit.score(X_test, Y_test)

# print the mean accuracy of testing predictions
print("DTC mean accuracy (Test) = " + str(round(test_score, 4)))

# Plot non-normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "4","5"])
# Plot normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "4","5"], 	normalise=True)


# In[67]:


from collections import  Counter
Counter(predicted)


# In[10]:


from sklearn.linear_model import LogisticRegression as LREG

# a logistic regression model with default values
lreg = LREG()
# fit the model using some training data
lreg_fit = lreg.fit(X_train, Y_train)
# generate a mean accuracy score for the training data
train_score = lreg.score(X_train, Y_train)
# print the R2 of training data
print("Logistic regression score (Train) = " + str(round(train_score, 4)))


# In[11]:


from sklearn.model_selection import GridSearchCV

# tune the hyperparameters for the logistic regression model
tuned_parameters = [{'penalty': ['l1', 'l2'],
                     'random_state': [3, 5, None]}]

scores = ['accuracy', 'f1_macro', 'recall_macro']

for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(LREG(), tuned_parameters, cv=5,
                       scoring= score)
    clf.fit(X_train, Y_train)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")


# In[12]:


# check the accuracy metrics again
# a logistic regression model with specific values
lreg = LREG(penalty = 'l2', random_state = 3)
# fit the model using some training data
lreg_fit = lreg.fit(X_train, Y_train)
# generate a mean accuracy score for the predicted data
train_score = lreg.score(X_train, Y_train)
# print the R2 of training data
print("Logistic regression score (after tuning hyperparameters)(Train) = " + str(round(train_score, 4)))


# In[13]:


# predict the test data
predicted = lreg_fit.predict(X_test)

# generate a mean accuracy score for the predicted data
test_score = lreg_fit.score(X_test, Y_test)

# print the R2 of testing predictions
print("Logistic regression R2 (Test) = " + str(round(test_score, 4)))

# Plot non-normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"])

# Plot normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"], 	normalise=True)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier as KNN

# a KNN model with default values
knn = KNN()
knn_fit = knn.fit(X_train, Y_train)
# generate an accuracy score for the predicted data
train_score = knn.score(X_train, Y_train)
# print the R2 of training data
print("KNN accuracy score(Train) = " + str(round(train_score, 4)))


# In[16]:


from sklearn.model_selection import GridSearchCV

# tune the hyperparameters for the logistic regression model
tuned_parameters = [{'n_neighbors': [1, 3, 5],
                     'algorithm': ['auto', 'ball tree'],
                    'leaf_size': [10, 30, 50]}]

scores = ['accuracy', 'f1_macro', 'recall_macro']

for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(KNN(), tuned_parameters, cv=3,
                       scoring= score)
    clf.fit(X_train, Y_train)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")


# In[17]:


# check the accuracy metrics again
# a logistic regression model with specific values
knn = KNN(algorithm = 'auto', leaf_size = 10, n_neighbors = 1)
# fit the model using some training data
knn_fit = knn.fit(X_train, Y_train)
# generate a mean accuracy score for the predicted data
train_score = knn_fit.score(X_train, Y_train)
# print the score of training data
print("KNN accuracy score(Train) (after tuning hyperparameters)(Train) = " + str(round(train_score, 4)))

# predict the test data
predicted = knn_fit.predict(X_test)

# generate an accuracy score for the predicted data
test_score = knn.score(X_test, Y_test)
# print the score of testing predictions
print("KNN accuracy score(Test) = " + str(round(test_score, 4)))

# Plot non-normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"])
# Plot normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"], 	normalise=True)


# In[18]:


from sklearn.ensemble import VotingClassifier 

# model essemble based on hard voting
hvoting_clf = VotingClassifier(estimators=[
    ('lreg', lreg),
    ('knn', knn),
    ('dtc',dtc)],voting='hard')
# predict the test data
hvoting_fit = hvoting_clf.fit(X_train, Y_train)
# generate a mean accuracy score for the predicted data
test_score = hvoting_fit.score(X_test, Y_test)
scores_test["Hard voting"] = round(test_score, 4)

# print the accuracy score of testing predictions
print("Hard voting accuracy score (Test) = " + str(round(test_score, 4)))


# In[19]:


from sklearn.ensemble import VotingClassifier 

# model essemble based on soft voting
svoting_clf = VotingClassifier(estimators=[
    ('lreg', lreg),
    ('knn', knn),
    ('dtc',dtc)],voting='soft')

# predict the test data
svoting_fit = svoting_clf.fit(X_train, Y_train)
# generate a mean accuracy score for the predicted data
test_score = svoting_fit.score(X_test, Y_test)
scores_test["Soft voting"] = round(test_score, 4)

# print the accuracy score of testing predictions
print("Soft voting accuracy score (Test) = " + str(round(test_score, 4)))


# In[21]:


# print the all the score of the modelds
print("The acccuracy score of testing predictions of each is: ")
for score in scores_test:
    print(score + ": " + str(scores_test[score]))
print("\n")
# pick the best model with the highest accuracy score 
best_model = max(scores_test, key = scores_test.get)
print("Therefore, the best model is: " + best_model 
      + ", the accuracy score is: " + str(scores_test[best_model]))


# In[29]:


from sklearn.ensemble import AdaBoostClassifier

# model enssemble based on baggingclassifier
boosting_lreg_clf = AdaBoostClassifier(base_estimator = lreg, learning_rate = 2, random_state=0)

# predict the test data
boosting_lreg_fit = boosting_lreg_clf.fit(X_train, Y_train)

# predict the test data
predicted = boosting_lreg_fit.predict(X_test)
# generate a mean accuracy score for the predicted data
test_score = boosting_lreg_fit.score(X_test, Y_test)

# print the accuracy score of testing predictions
print("Bagging of losgistic accuracy score (Test) = " + str(round(test_score, 4)))


# In[32]:


from sklearn.ensemble import BaggingClassifier

# model enssemble based on baggingclassifier
bagging_lreg_clf = BaggingClassifier(base_estimator = lreg,n_estimators=10, random_state=0)

# predict the test data
bagging_lreg_fit = bagging_lreg_clf.fit(X_train, Y_train)

# predict the test data
predicted = bagging_lreg_fit.predict(X_test)
# generate a mean accuracy score for the predicted data
test_score = bagging_lreg_fit.score(X_test, Y_test)

# print the accuracy score of testing predictions
print("Bagging of losgistic accuracy score (Test) = " + str(round(test_score, 4)))


# In[38]:


from sklearn.model_selection import GridSearchCV

# tune the hyperparameters for the logistic regression model
tuned_parameters = [{'learning_rate': [1, 3, 5],
                     'random_state': [10,30]}]

scores = ['accuracy', 'f1_macro', 'recall_macro']

for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=3,
                       scoring= score)
    clf.fit(X_train, Y_train)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")


# In[40]:


# check the accuracy metrics again
# a boosting of logistic regression model with specific values
boosting_lreg = AdaBoostClassifier(base_estimator = lreg, learning_rate = 1, random_state = 10)
# fit the model using some training data
boosting_lreg_fit = boosting_lreg.fit(X_train, Y_train)
# generate a mean accuracy score for the predicted data
train_score = boosting_lreg_fit.score(X_train, Y_train)
# print the score of training data
print("Boosting for LREG accuracy score(Train) (after tuning hyperparameters)(Train) = " + str(round(train_score, 4)))

# predict the test data
predicted = boosting_lreg_fit.predict(X_test)

# generate an accuracy score for the predicted data
test_score = boosting_lreg_fit.score(X_test, Y_test)
# print the score of testing predictions
print("Boosting for LREG accuracy score(Test) = " + str(round(test_score, 4)))

# Plot non-normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"])
# Plot normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"], 	normalise=True)


# In[41]:


# check the accuracy metrics again
# a boosting of DTC model with specific values
boosting_dtc = AdaBoostClassifier(base_estimator = dtc, learning_rate = 1, random_state = 10)
# fit the model using some training data
boosting_dtc_fit = boosting_dtc.fit(X_train, Y_train)
# generate a mean accuracy score for the predicted data
train_score = boosting_dtc_fit.score(X_train, Y_train)
# print the score of training data
print("Boosting for DTC accuracy score(Train) (after tuning hyperparameters)(Train) = " + str(round(train_score, 4)))

# predict the test data
predicted = boosting_dtc_fit.predict(X_test)

# generate an accuracy score for the predicted data
test_score = boosting_dtc_fit.score(X_test, Y_test)
# print the score of testing predictions
print("Boosting for DTC accuracy score(Test) = " + str(round(test_score, 4)))

# Plot non-normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"])
# Plot normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"], 	normalise=True)


# In[50]:


from sklearn.ensemble import VotingClassifier 

# model essemble based on soft voting
svoting_clf = VotingClassifier(estimators=[
    ('lreg', boosting_lreg),
    ('knn', knn),
    ('dtc',boosting_dtc)],voting='soft')

# predict the test data
svoting_fit = svoting_clf.fit(X_train, Y_train)
# generate a mean accuracy score for the predicted data
test_score = svoting_fit.score(X_test, Y_test)
scores_test["Soft voting(boosting)"] = round(test_score, 4)

# print the accuracy score of testing predictions
print("Soft voting accuracy score (Test) = " + str(round(test_score, 4)))

