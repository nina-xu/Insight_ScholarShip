# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:54:58 2019

@author: Ning

Scrape information regarding financial aid awarded to international students
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import re
import time
import numpy as np
import pandas as pd

driver = webdriver.Chrome()

# The universities are organized by states on this website
# I copied the portion of the html that contains state initials and names
# into a txt file
with open('Data/states.txt','r') as file:
    states = file.readlines()

def get_state_name(state):
    """
    Extract state initial and name from the html
    """
    state_init = state[19:21]
    state_name = re.search('>\D+<',state).group()[1:-1]
    return state_init, state_name    


def get_intl_awarded(text, total_intl):
    """
    The number of int'l students receiving financial aid is represented in
    different formats for each institution. 
    This function converts different formats into counts
    """
    if text == 'All':
        text = '100%'
    if re.search('%',text):
        pct = int(re.sub('%','',text))
        intl_awarded = int(total_intl * pct /100)
    elif text == '':
        intl_awarded = 0
    else:
        intl_awarded = int(text)
    return intl_awarded

# record entries in one line of the table
def get_entry(line):
    """
    Scrape one line of the table.
    The columns are:
        total annual cost; 
        # of int'l students; 
        # awarded aid; 
        Average aid
    """
    entry = {}
    entry['state_init'] = state_init
    entry['state_name'] = state_name
    entry['inst_name'] = line.td.text.strip()
    entry['link'] = line.td.a['href']
    entries = line.find_all('td')
    annual_cost_raw = entries[1].text[1:-3]
    entry['annual_cost'] = int(re.sub(',','',annual_cost_raw))
    total_intl = int(re.sub(',','',entries[2].text))
    entry['total_intl'] = total_intl
    try:
        entry['intl_awarded'] = get_intl_awarded(entries[3].text, total_intl)
    except:
        entry['intl_awarded'] = np.NaN
        print(line, 'something went wrong')
    avg_award_raw = line.find_all('td')[4].text[1:-3]
    entry['avg_award'] = int(re.sub(',','',avg_award_raw))
    return(entry)
    
def get_state(state_init, state_name):
    """
    Scrape info of all institutions in a given state
    """
    state_name_link = re.sub(' ', '%20', state_name)
    url = (
            'https://www.internationalstudent.com/schools_awarding_aid/'
            +state_init
            +'/'
            +state_name_link
            +'.html'
            )
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml') 
    table = soup.find('table')
    lines = table.find_all('tr')
    entries = []
    if len(lines) > 2:
        for line in lines[1:]:
            entries.append(get_entry(line))
    elif len(lines) == 2:
        if lines[1].text == 'No results found.':
            return(entries)
        else: # it means there is one school
            entries.append(get_entry(lines[1]))
    else:
        print('something went wrong', state_name)
    return(entries)



all_institutions = []
for state in states:
    state_init, state_name = get_state_name(state)
    inst_info = get_state(state_init, state_name)
    all_institutions.extend(inst_info)
    time.sleep(np.random.lognormal(0.75) + 0.2)
    

institution_info = pd.DataFrame(all_institutions)
institution_info.to_csv('Data/InternationalStudent_com.csv', index = False)