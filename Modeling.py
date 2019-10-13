# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:27:44 2019

@author: Ning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# import data
data = pd.read_csv("Data/data 258.csv")

features = ["GRNRALT_rate", "TUITION2", "diverse_ind", "avg_award", "pct_awarded"]
#%% EDA
all_vars = [
    "GRNRALT_rate",
    "TUITION2",
    "diverse_ind",
    "avg_award",
    "pct_awarded",
    "intl_pct",
]

# plot the distributions of my features
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.hist(data[features[i]])
    plt.xlabel(features[i])
# plot the correlations between features
plt.figure(figsize=(20, 20))
c = 0
for i in range(len(all_vars)):
    for j in range(len(all_vars)):
        c += 1
        plt.subplot(6, 6, c)
        plt.scatter(data[all_vars[i]], data[all_vars[j]])
        plt.xlabel(all_vars[i])
        plt.ylabel(all_vars[j])

#%% train test split
np.random.seed(922)
train, test = train_test_split(data, test_size=0.2)

#%% Binomial regression
features_std = []
for i in range(len(features)):
    features_std.append(features[i] + "_std")
# standardize the features first
feat_normalize = scale(train[features], axis=0)
for i in range(len(features)):
    data.loc[:, features[i] + "_std"] = feat_normalize[:, i]

X = sm.add_constant(train[features_std], prepend=False)
binomial_scholarship = sm.GLM(
    train[["EFYNRALT", "EFYUS"]], X, family=sm.families.Binomial()  # [success, failure]
)
result = binomial_scholarship.fit()
print(result.summary())

# get prediction
pred_test = result.predict(test[features])
pred_train = result.predict(train[features])

# make plot
plt.subplot(1, 2, 1)
plt.scatter(pred_train, train["intl_pct"] / 100)
plt.title("train")
plt.xlabel("y hat")
plt.ylabel("y")
plt.subplot(1, 2, 2)
plt.scatter(pred_test, test["intl_pct"] / 100)
plt.title("test")
plt.xlabel("y hat")
plt.ylabel("y")

#%% linear regression with logit transformation
logit = np.log(data.intl_pct / (100 - data.intl_pct))
data.loc[:, "logit_intl_pct"] = logit

scaler = StandardScaler()
scaler.fit(train[features])
transformed_linear = LinearRegression()
result2 = transformed_linear.fit(X=scaler.transform(train[features]), y=train.logit)
y_pred = result2.predict(X=scaler.transform(test[features]))

# plot the result
y_pred_orig = np.exp(y_pred) / (1 + np.exp(y_pred))
plt.scatter(y_pred_orig * 100, test.intl_pct)
plt.plot([0, 20], [0, 20])
plt.title("Actual vs. Predicted Enrollment")
plt.xlabel("% Enrolled (Predicted)")
plt.ylabel("% Enrolled (Actual)")
