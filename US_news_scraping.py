# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:41:26 2019

@author: Ning

Scrape the insitution name, ranking, and tuition 
from the US News list of best national universities
"""


from selenium import webdriver
import time
import numpy as np

driver = webdriver.Chrome()
driver.get('https://www.usnews.com/best-colleges/rankings/national-universities')

# obtain insitution name, ranking, and tuition based on xpath
def get_institution_name(i, path_div=1):
    path = (
            '//*[@id="school-' 
            + str(i) 
            + '"]/div/div[' 
            + str(path_div) 
            + ']/div[1]/div[1]/h3/a'
            )
    institution = driver.find_elements_by_xpath(path)
    name = institution[0].text
    return(name)

def get_ranking(i, path_loc = 1):
    path = (
            '//*[@id="school-' 
            + str(i) 
            + '"]/div/div[' 
            + str(path_loc) 
            + ']/div[1]/div[2]/ul/li/a/div/strong'
            )
    element = driver.find_elements_by_xpath(path)
    ranking = element[0].text
    return(ranking)

def get_tuition(i, path_loc = 1):
    path = (
            '//*[@id="school-' 
            + str(i) 
            + '"]/div/div[' 
            + str(path_loc) 
            + ']/div[2]/div[1]/dl/div/dd'
            )
    element = driver.find_elements_by_xpath(path)
    tuition = element[0].text
    return(tuition)

# interact with the webpage to load more institutions
# scroll down to the bottom of the page and the page automatically reloads
def load_more():
    driver.execute_script("window.scrollBy(0,document.body.scrollHeight)")
    time.sleep(np.random.normal(20,0.45))
# click on the reload button to reload
def load_more_with_click():
    #driver.execute_script("window.scrollBy(0,document.body.scrollHeight)")
    path = '//*[@id="search-content"]/div/div[1]/div[2]/div[3]/div/div[2]'
    button = driver.find_element_by_xpath(path)
    button.click()
    time.sleep(np.random.normal(20,0.25) + 0.2)

# scrape the university names
# this code runs into errors every once in a while, 
# but I was able to help by manually scrolling/clicking
USNews = []
for i in range(381):
    
    try:
        USNews.append(get_institution_name(i))
    except:
        print(str(i))
        try:
            USNews.append(get_institution_name(i, path_div = 2))
        except:
            load_more()
            try:
                USNews.append(get_institution_name(i))
            except:
                load_more_with_click()
                try:
                    USNews.append(get_institution_name(i))
                except:
                    USNews.append()

# save list of names to file
with open('Data/US News 2020.txt', 'w') as filehandle:
    for listitem in USNews:
        filehandle.write('%s\n' % listitem)

# scrape the rankings
rankings = []
for i in range(381):
    try:
        rankings.append(get_ranking(i))
    except:
        try:
            rankings.append(get_ranking(i, path_loc = 2))
        except:
            rankings.append('NaN')
with open('Data/US News 2020 rankings.txt', 'w') as filehandle:
    for listitem in rankings:
        filehandle.write('%s\n' % listitem)

# scrape the tuitions
tuitions = []
for i in range(381):
    try:
        tuitions.append(get_tuition(i))
    except:
        try:
            tuitions.append(get_tuition(i, path_loc = 2))
        except:
            tuitions.append('NaN')
with open('Data/US News 2020 tuitions.txt', 'w') as filehandle:
    for listitem in tuitions:
        filehandle.write('%s\n' % listitem)