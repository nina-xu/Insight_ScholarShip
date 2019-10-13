# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:29:14 2019

@author: Ning
"""

import pandas as pd
import numpy as np
import re

#%% load and merge IPEDS data
# https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx
universities = pd.read_csv("Data/hd2017.csv", encoding="latin-1")
enrollment = pd.read_csv("Data/effy2017.csv")
charges = pd.read_csv("Data/ic2017_ay.csv")
adm = pd.read_csv("Data/adm2017.csv")
ef2017b = pd.read_csv("Data/ef2017b.csv")
ef2017c = pd.read_csv("Data/ef2017c.csv")

# filter only enrollment data for 4-year degrees
enrollment_4yr = enrollment[enrollment.EFFYLEV == 2]  # (6478, 33)
# over 50% universities don't have an international enrollment, so remove
# further filter only collges who have a >0 international student enrollment
enrollment_4yr_intl = enrollment_4yr[enrollment_4yr.EFYNRALT > 0]  # (3221, 33)

# merge university characteristics and enrollment data
data = pd.merge(enrollment_4yr_intl, universities, how="left", on="UNITID")
data = pd.merge(data, charges, how="left", on="UNITID")
data = pd.merge(data, adm, how="left", on="UNITID")
data = pd.merge(data, ef2017b, how="left", on="UNITID")
data = pd.merge(data, ef2017c, how="left", on="UNITID")

#%% load US News data and merge

## load
# institution name
names = []
with open("Data/US News 2020.txt", "r") as file:
    for line in file:
        name = line[:-1]
        # replace--  with - in the name
        name = re.sub("--", "-", name)
        names.append(name)
# institution ranking
rankings = []
with open("Data/US News 2020 rankings.txt", "r") as file:
    for line in file:
        rankings.append(line[:-1])
# tuition
tuitions = []
with open("Data/US News 2020 tuitions.txt", "r") as file:
    for line in file:
        tuitions.append(line[:-1])

## clean
# convert ranking to numeric
rankings_clean = []
for i in range(292):
    rankings_clean.append(rankings[i][1:])
for i in range(292, len(rankings)):
    rankings_clean.append("293")
rankings_clean = [int(ranking) for ranking in rankings_clean]
# convert tuition to numeric
tuitions_clean = [re.sub("\(out-of-state\)", "", t) for t in tuitions]
tuitions_clean = [re.sub(",", "", t) for t in tuitions_clean]
for i in range(len(tuitions)):
    if tuitions_clean[i] == "N/A":
        tuitions_clean[i] = np.nan
    else:
        tuitions_clean[i] = int(tuitions_clean[i][1:])

## merge
# split column of aliases in preparation for searching
aliases = []
for al in data.IALIAS:
    als = re.split("\|", al)
    if len(als) == 1:  # if not split by |
        als = re.split(",", als[0])
    if len(als) == 1:
        als = re.split(";", als[0])
    if len(als) == 1:
        als = re.split("/", als[0])
    # remove white spaces at beginning and end
    als_no_space = [al.strip() for al in als]
    aliases.append(als_no_space)

# for each university in the US News list, find a match from the names/aliases
# this is a pain by the way
match = [0] * len(names)
official_names = []
for i in range(len(names)):
    off_name = ""
    if names[i] in list(data.INSTNM):  # found a match
        off_name = names[i]
        match[i] = 1
    elif ("The " + names[i]) in list(data.INSTNM):  # "The University of ...."
        off_name = "The " + names[i]
        match[i] = 1
    else:
        for j in range(len(data.INSTNM)):
            if names[i] in aliases[j]:
                off_name = data.INSTNM[j]
                match[i] = 1
                break
    official_names.append(off_name)

# second round of matching
# for both lists, remove: at, -, 'Main Campus'
instnm_2 = [re.sub(" at ", " ", name) for name in data.INSTNM]
instnm_2 = [re.sub("-", " ", name) for name in instnm_2]
instnm_2 = [re.sub("Main Campus", "", name) for name in instnm_2]
instnm_2 = [re.sub("\.", "", name) for name in instnm_2]
instnm_2 = [name.strip() for name in instnm_2]

names_2 = [re.sub("-", " ", name) for name in names]
names_2 = [re.sub("\.", "", name) for name in names_2]

# round 2
for i in range(len(names_2)):
    if match[i] == 0:
        if names_2[i] in instnm_2:
            match[i] = 1
            official_names[i] = data.INSTNM[
                np.where(np.array(instnm_2) == names_2[i])[0][0]
            ]
        elif ("The " + names_2[i]) in instnm_2:
            match[i] = 1
            official_names[i] = data.INSTNM[
                np.where(np.array(instnm_2) == ("The " + names_2[i]))[0][0]
            ]

# third round, manual matching
manual = pd.read_csv("Data/manual matching.csv")
for i in range(len(manual.names_3)):
    ind = np.where(np.array(names_2) == manual.names_3[i])[0][0]
    official_names[ind] = manual.instnm_3[i]
    if manual.instnm_3[i] != "":
        match[ind] = 1

#%% Retention and completion rates
# 6-year graduation rates 2017
gr = pd.read_csv("Data/gr2017.csv")
gr_4yr = gr[gr.GRTYPE == 3]
gr_cohort = gr[gr.GRTYPE == 2]
data = pd.merge(data, gr_4yr, how="left", on="UNITID")

# calculate the rate
# get a list of column names I want to calculate rates for
var_names = gr_4yr.columns[7:]

cal_names = []
for var_name in var_names:
    m = re.search("\AGR\w+", var_name)
    if m:
        cal_names.append(m.group(0))
print(cal_names)
cal_names[-1] = "GRNRALW "

for name in cal_names:
    var2 = name + "_y"  # cohort
    var3 = name + "_x"  # completers
    var_rate = name + "_rate"
    data[var_rate] = data[var3] / data[var2]
    data[var_rate][data[var3] == 0] = 0

#%% adding internationalstudent.com
scholarship = pd.read_csv("Data/InternationalStudent_com.csv")

# matching with url
url1 = data.WEBADDR
url1_cleaned = []
for url in url1:
    url = url.lower()
    url = re.sub("http://", "", url)
    url = re.sub("https://", "", url)
    url = re.sub("www.", "", url)
    url = re.sub("/", "", url)
    url1_cleaned.append(url)
data.loc[:, "WEBADDR_cleaned"] = url1_cleaned

url2 = scholarship.link
url2_cleaned = []
for url in url2:
    url = url.lower()
    url = re.sub("http://", "", url)
    url = re.sub("https://", "", url)
    url = re.sub("http//", "", url)
    url = re.sub("www.", "", url)
    url = re.sub(".edu\D+", ".edu", url)
    url = re.sub("admissions.", "", url)
    url2_cleaned.append(url)
scholarship.loc[:, "link_cleaned"] = url2_cleaned

# merge
data = pd.merge(
    data, scholarship, how="left", left_on="WEBADDR_cleaned", right_on="link_cleaned"
)
#%%
data.to_csv("Data/Data Merged.csv", index=False)
