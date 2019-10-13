# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:10:32 2019

@author: Ning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from IPython.display import Image 
from subprocess import call

data = pd.read_csv('Data/data 258.csv')
features = ['GRNRALT_rate','TUITION2','diverse_ind','avg_award','pct_awarded']

np.random.seed(951)
train, test = train_test_split(data, test_size = 0.2)
transformed_tree = DecisionTreeRegressor(random_state=951)
result2 = transformed_tree.fit(X = train[features], y = train.logit)

def reverse_logit(logit):
    return np.exp(logit)/(1 + np.exp(logit))

y_pred_logit = result2.predict(X = test[features])
y_pred = reverse_logit(y_pred_logit)
plt.scatter(y_pred*100, test.intl_pct)
plt.title('Actual vs. Predicted Enrollment')
plt.xlabel('% Enrolled (Predicted)')
plt.ylabel('% Enrolled (Actual)')
plt.plot([0,40],[0,40])

# mean absolute error
np.mean(np.abs(y_pred*100 -test.intl_pct)) # 5.69%

# visualize
export_graphviz(transformed_tree, out_file='tree.dot', feature_names = features,
                filled=True, rounded=True)
# Convert to png using system command (requires Graphviz)
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
Image(filename = 'tree.png')