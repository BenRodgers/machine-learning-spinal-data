#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:50:21 2019

@author: benrodgers
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../Data/Dataset_spine.csv")

df.info()
df.head()

df = df.drop("Unnamed: 13", 1)

df.columns = ['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle', 'sacral_slope',
        'pelvic_radius', 'degree_spondy', 'pelvic_slope', 'direct_tilt', 'thoracic_slope',
        'cervical_tilt','sacrum_angle','scoliosis_slope','target']


'''
Explore the Data
'''

df.info()
df.head()
df.describe()


'''
Look for outliers
'''
plt.figure(figsize=(10,10))
sns.set(style="whitegrid")
sns.boxplot(data = df, palette="Set3")
plt.show()


'''
Remove outliers
'''

df.drop(df[(df['pelvic_incidence'] > df['pelvic_incidence'].quantile(0.95)) | (df['pelvic_incidence'] < df['pelvic_incidence'].quantile(0.00))].index, inplace=True)
df.drop(df[(df['pelvic_tilt'] > df['pelvic_tilt'].quantile(0.91)) | (df['pelvic_tilt'] < df['pelvic_tilt'].quantile(0.0))].index, inplace=True)
df.drop(df[(df['pelvic_radius'] > df['pelvic_radius'].quantile(0.96)) | (df['pelvic_radius'] < df['pelvic_radius'].quantile(0.00))].index, inplace=True)
df.drop(df[(df['lumbar_lordosis_angle'] > df['lumbar_lordosis_angle'].quantile(0.95)) | (df['lumbar_lordosis_angle'] < df['lumbar_lordosis_angle'].quantile(0.0))].index, inplace=True)
df.drop(df[(df['degree_spondy'] > df['degree_spondy'].quantile(0.90)) | (df['degree_spondy'] < df['degree_spondy'].quantile(0.10))].index, inplace=True)
df = df.drop('lumbar_lordosis_angle', axis=1)


'''
Look for outliers
'''
plt.figure(figsize=(20,20))
sns.set(style="whitegrid")
sns.boxplot(data = df, palette="Set3")
plt.show()


'''
Heatmaps show the correlations
'''
plt.figure(figsize=(10,10))
corr = df.corr()
colormap = sns.diverging_palette(220, 10, as_cmap=True)
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()


'''
Label encode the final column (Normal, abnormal -> 0,1)
'''
from sklearn import preprocessing
label = preprocessing.LabelEncoder()
df['target'] = label.fit_transform(df.target.values)


'''
Split the data for test and training
'''
from sklearn.model_selection import train_test_split
y = df['target'].values
x_data = df.drop(['target'], axis = 1)
# Test/Train split
x_train, x_test, y_train, y_test = train_test_split(x_data,y,test_size = 0.30,random_state=0)


#Random Forest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, random_state = 1)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print("Accuracy of Random Forest Classifier {:.9f}".format(accuracy_score(pred, y_test)))


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print("Accuracy of Naive Bayes {:.9f}".format(accuracy_score(pred, y_test)))

#kNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print("Accuracy of kNN {:.9f}".format(accuracy_score(pred, y_test)))


#Gradient Boost
from xgboost import XGBClassifier
model = XGBClassifier()
pred = model.fit(x_train, y_train)
# make predictions for test data
pred = model.predict(x_test)

from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(12, 6))
plot_importance(model, ax=ax)
print("Accuracy of Gradient Boosting XGBoost {:.9f}".format(accuracy_score(pred, y_test)))
