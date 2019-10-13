# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:05:31 2019

@author: Ning
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

data = pd.read_csv("Data/data 289.csv")

#%% calculate one-on-one correlations between each potential feature
# and the target variable
all_vars = list(data.columns)
all_vars.remove("intl_pct")

cors = []
for var in all_vars:
    try:
        if data[var].dtypes in ("int64", "float64"):
            d_tmp = data[["intl_pct", var]].dropna()
            if np.var(d_tmp[var]) > 0:
                cor, p = pearsonr(d_tmp.intl_pct, d_tmp[var])
                cors.append({
                        "var": var, 
                        "cor": cor, 
                        "p": p, 
                        "n": d_tmp.shape[0]
                        })
    except:
        print(var)

# select variables that have larger than 0.3 correlation
cors_df = pd.DataFrame(cors)
vars_select = cors_df[cors_df.cor > 0.3]
vars_select = vars_select[vars_select.n > 80]
print(vars_select)

vars_select = cors_df[cors_df.cor < -0.3]
vars_select = vars_select[vars_select.n > 80]
print(vars_select)

#%% create variables
# international student enrollment percentage
intl_pct = data["EFYNRALT"] / data["EFYTOTLT"] * 100

# diversity index within the US students
# calculate total of US students
my_total = (
    data["EFYAIANT"]
    + data["EFYASIAT"]
    + data["EFYBKAAT"]
    + data["EFYHISPT"]
    + data["EFYWHITT"]
    + data["EFYNHPIT"]
    + data["EFY2MORT"]
)
prop_AIAN = data["EFYAIANT"] / my_total
prop_ASIA = data["EFYASIAT"] / my_total
prop_BKAA = data["EFYBKAAT"] / my_total
prop_HISP = data["EFYHISPT"] / my_total
prop_WHIT = data["EFYWHITT"] / my_total
prop_NHPI = data["EFYNHPIT"] / my_total
prop_2MOR = data["EFY2MORT"] / my_total

diversity_index = 1 - (
    np.square(prop_AIAN)
    + np.square(prop_ASIA)
    + np.square(prop_BKAA)
    + np.square(prop_HISP)
    + np.square(prop_WHIT)
    + np.square(prop_NHPI)
    + np.square(prop_2MOR)
)

data.loc[:, "diverse_ind"] = diversity_index
data.loc[:, "intl_pct"] = intl_pct

# calculate number of US students for binomial regression
data.loc[:,'EFYUS'] = data['EFYTOTLT'] - data['EFYNRALT']

data.to_csv("Data/Data Merged.csv", index=False)