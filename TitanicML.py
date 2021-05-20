import pandas as pd
import numpy as np
from numpy import mean
import csv
import random
import matplotlib.pyplot as plt

from random import seed

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Imputing missing values and scaling values
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import cv

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

from scipy.stats import randint as sp_randint

# Internal ipython tool for setting figure size
from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns

sns.set(font_scale=2)
data = pd.read_csv('Titanic.csv')
test_df = pd.read_csv('Titanic_test.csv')
combine = [data, test_df]

"""
for col in list(data.columns):
    if 'TotalCharges' in col or 'MonthlyCharges' in col or 'tenure' in col:
        data[col] = data[col].astype(float)
"""

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# Get the columns with > 50% missing
missing_df = missing_values_table(data);
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
print('We will remove %d columns.' % len(missing_columns))

print(
    data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# grid = sns.FacetGrid(data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()

# grid2 = sns.FacetGrid(data, row = 'Embarked', size = 2.2, aspect = 1.6)
# grid2.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid2.add_legend()
# plt.show()

# Result is that Class, Sex and Embarked are interesting for ML

data = data.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [data, test_df]

for dataset in combine:
    # dataset = dataset.replace(r'^\s+$', np.nan, regex=True)  # IMPORTANT to replace empty cells
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Create new feature from name

# print(pd.crosstab(data['Title'], data['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# print(data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

data = data.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [data, test_df]

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}  # convert to ordinal
sex_mapping = {"male": 0, "female": 1}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

# Complete missing data - for age a numerically continuous feature
# Note correlation between Gender, Pclass and Age

guess_ages = data.groupby(['Sex', 'Pclass'])['Age'].apply(np.mean)
# print(data.loc[{(data['Sex'] == 1) and (data['Pclass'] == 2)},'Age'])

"""
    for i in range(0,2):
        for j in range (0,3):
            guess_data = dataset[(dataset['Sex'] == i) and (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_data.median()
            guess_ages[i,j] = int(age_guess/0.5 + 0.5)*0.5
"""

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex'] == i) & (dataset['Pclass'] == j + 1), 'Age'] = \
            guess_ages[i, j + 1]

"""
data['AgeBand'] = pd.cut(data['Age'], 5)
data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

data = data.drop(['AgeBand'], axis=1)
combine = [data, test_df]
for dataset in combine:
    dataset['Age'] = dataset['Age'].astype(int)
"""

for dataset in combine:
    dataset['sqrt_' + 'Age'] = np.sqrt(dataset['Age'])

data = data.drop(['Age'], axis=1)
test_df = test_df.drop(['Age'], axis=1)
combine = [data, test_df]

# Combine family features for a simpler classification of whether alone or not

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
# ascending=False)
# print(data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

data = data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [data, test_df]

# Completing a categorical feature - embarked has a few missing - fill with most common

freq_port = data['Embarked'].dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

port_mapping = {'S': 0, 'C': 1, 'Q': 2}
data['Embarked'] = data['Embarked'].map(port_mapping)
test_df['Embarked'] = test_df['Embarked'].map(port_mapping)
print(data.head())

# Features selection for ML

test_df.loc[test_df['Fare'].isnull(), 'Fare'] = 1
combine = [data, test_df]
for dataset in combine:
    dataset.loc[(dataset['Fare'] <= 1), 'Fare'] = 1
    # dataset['sqrt_' + 'Fare'] = np.sqrt(dataset['Fare'])
    dataset['log_' + 'Fare'] = np.log(dataset['Fare'])

data = data.drop(['Fare'], axis=1)
test_df = test_df.drop(['Fare'], axis=1)

# After Feature Importance
data = data.drop(['Embarked', 'IsAlone'], axis=1)
test_df = test_df.drop(['Embarked', 'IsAlone'], axis=1)
combine = [data, test_df]

# categorical_subset = pd.get_dummies(data['Embarked'])
# features = pd.concat([data, categorical_subset], axis=1)
# features.info()

correlations = data.corr()['Survived'].sort_values()
print(correlations.head(15))

X = data.drop(['Survived'], axis=1)
y = pd.DataFrame(data['Survived'])
X_test = test_df.drop("PassengerId", axis=1).copy()
print(X.describe())
print(y.shape)
print(X_test.describe())

# Convert y to one-dimensional array (vector)
y = np.array(y).reshape((-1,))

# No need for further scaling - no units aside from Fare Function to calculate mean absolute error, way to compare
# different ML methods Our problem is a classification and regression problem. We want to identify relationship
# between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a
# category of machine learning which is called supervised learning as we are training our model with a given dataset.

svc = SVC()
svc.fit(X, y)
# Y_pred = knn.predict(X_test)
acc_svc = round(svc.score(X, y) * 100, 2)
print('SV Classification: %0.4f' % acc_svc)

knn = KNeighborsClassifier(metric='euclidean', n_neighbors=3)
knn.fit(X, y)
# Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X, y) * 100, 2)
print('K-Nearest Neighbors Classification: %0.4f' % acc_knn)
repeatedkfold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
cv_results = cross_val_score(knn, X, y, cv=repeatedkfold, scoring='accuracy')
print(mean(cv_results))

gnb = GaussianNB()
gnb.fit(X, y)
# Y_pred = knn.predict(X_test)
acc_gnb = round(gnb.score(X, y) * 100, 2)
print('Naive Bias Classification: %0.4f' % acc_svc)

random_forest = RandomForestClassifier(max_features='log2', n_estimators=1000)
random_forest.fit(X, y)
# Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X, y) * 100, 2)
print('Random Forest Classifier: %0.4f' % acc_random_forest)
repeatedkfold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
cv_results = cross_val_score(random_forest, X, y, cv=repeatedkfold, scoring='accuracy')
print(mean(cv_results))

gradient_boosted = GradientBoostingClassifier(n_estimators=100)
gradient_boosted.fit(X, y)
# Y_pred = gradient_boosted.predict(X_test)
acc_gradient_boosted = round(gradient_boosted.score(X, y) * 100, 2)
print('Gradient Boosted Classification: %0.4f' % acc_gradient_boosted)
repeatedkfold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
cv_results = cross_val_score(gradient_boosted, X, y, cv=repeatedkfold, scoring='accuracy')
print(mean(cv_results))


# K-Nearest Neighbors Classification: 87.9900
# [0.77094972 0.80898876 0.76966292 0.8258427  0.81460674]
# SV Classification: 82.6000
# [0.8547486  0.87078652 0.79213483 0.79775281 0.79775281]
# Naive Bias Classification: 82.6000
# [0.72067039 0.79213483 0.74157303 0.78089888 0.7752809 ]
# Random Forest Classifier: 91.6900
# [0.81564246 0.83146067 0.83707865 0.85393258 0.78089888]
# Gradient Boosted Classification: 89.0000
# [0.88826816 0.8258427  0.79775281 0.81460674 0.84269663]

# Choose the best hyperparameters for a model through random search and cross validation
# Dictionary for various ML models
# SGD

loss = ['hinge', 'modified_huber', 'log']
penalty = ['l1','l2']
alpha= [0.0001,0.001,0.01,0.1]
l1_ratio= [0.15,0.05,.025]
max_iter = [1,5,10,100,1000,10000]
sgd_grid = dict(loss=loss,penalty=penalty,max_iter=max_iter,alpha=alpha,l1_ratio=l1_ratio)

# Ridge
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ridge_grid = dict(alpha=alpha)

# K-Nearest - Neighbors
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
knn_grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)

# Support Vector Classifier
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
svc_grid = dict(kernel=kernel,C=C,gamma=gamma)

# Bagging Classifier
n_estimators = [10, 100, 1000]
bag_grid = dict(n_estimators=n_estimators)

# Random Forest
n_estimators = [10, 100, 1000,10000]
max_features = ['sqrt', 'log2']
rf_grid = dict(n_estimators=n_estimators,max_features=max_features)

# Logistic Regression
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
lr_grid = dict(solver=solvers,penalty=penalty,C=c_values)

# LGB
class_weight = [None,'balanced']
boosting_type = ['gbdt', 'goss', 'dart']
num_leaves = [30,50,100,150] # list(range(30, 150)),
learning_rate = list(np.logspace(np.log(0.005), np.log(0.2), base = np.exp(1), num = 10)) #1000
lgg_grid = dict(class_weight=class_weight, boosting_type=boosting_type, num_leaves=num_leaves, learning_rate =learning_rate)

# Running randomized search on hyperparameters
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3,
                                random_state=1)
n_iter_search = 3
random_search = RandomizedSearchCV(random_forest, param_distributions=rf_grid, n_iter=n_iter_search, cv=cv)
random_search.fit(X, y)
print(random_search.best_estimator_)
improved = random_search.score(X,y)
print(improved)


"""
# Gradient Boosting Classifier
loss = ['deviance', 'exponential']
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
min_samples_leaf = [1, 2, 4, 6, 8]
min_samples_split = [2, 4, 6, 10]
max_features = ['auto', 'sqrt', 'log2', None]
# Define the grid of hyperparameters to search
hyperparameter_grid = {'loss': loss,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=gradient_boosted,
                               param_distributions=hyperparameter_grid,
                               cv=4, n_iter=25,
                               scoring='neg_mean_absolute_error',
                               n_jobs=-1, verbose=1,
                               return_train_score=True,
                               random_state=42)

random_cv.fit(X, y)
print(random_cv.best_estimator_)
"""


# After hyperparameter tuning, final model for cross validation score and extraction
final_model = GradientBoostingClassifier(loss='exponential', max_depth=2, max_features='sqrt', min_samples_leaf=6,
                                         min_samples_split=4, n_estimators=800)
kfold = KFold(n_splits=5, shuffle=True)
cv_results = cross_val_score(final_model, X, y, cv=kfold, scoring='accuracy')
print(cv_results)

final_model.fit(X, y)
Y_pred = final_model.predict(X_test)

# Extract the feature importances into a dataframe
feature_results = pd.DataFrame({'feature': list(X.columns),
                                'importance': final_model.feature_importances_})

# Show the top 10 most important
feature_results = feature_results.sort_values('importance', ascending = False).reset_index(drop=True)
print(feature_results.head(10))

submission = pd.DataFrame({
"PassengerId": test_df["PassengerId"],
"Survived": Y_pred})
submission.to_csv('Titanic_submission.csv', index=False)


# Feature Importance
final_model = RandomForestClassifier(n_estimators=100)
final_model.fit(X,y)
model_pred = final_model.predict(X_test)

# Extract the feature importances into a dataframe
feature_results = pd.DataFrame({'feature': list(X.columns),
                                'importance': final_model.feature_importances_})

# Show the top 10 most important
feature_results = feature_results.sort_values('importance', ascending = False).reset_index(drop=True)
print(feature_results.head(10))
