import os
import urllib
import re
import json
from bs4 import BeautifulSoup
import requests
import sqlite3
import numpy as np

# Create connection to database
# conn = sqlite3.connect('news.db')
conn = sqlite3.connect('news_testing.db')
c = conn.cursor()

##################################################################

# Delete tables if they exist
c.execute('DROP TABLE IF EXISTS "keywords";')
# c.execute('DROP TABLE IF EXISTS "quotes";')

# Create tables in the database and add data to it.
create_keywords_table_command = '''
CREATE TABLE IF NOT EXISTS keywords (
    date VARCHAR(255) NOT NULL,
    keyword VARCHAR(255) NOT NULL,
    frequency int NOT NULL,
    num_articles int NOT NULL
	);
'''

##################################################################

get_data_command = '''
SELECT A.Date, body, number_articles FROM (SELECT Date, body FROM articles) A LEFT JOIN (SELECT date, COUNT(DATE) number_articles FROM articles GROUP BY date) B ON A.date = B.date;
'''

c.execute(create_keywords_table_command)
conn.commit()

print('------------- data extracted from articles (news.db) -------------')

##################################################################

raw_data = {}
num_articles = {}

c.execute(get_data_command)
count = 0
for row in c:
    num_articles[row[0]] = row[2]
    count += 1
    if not row[0] in set(raw_data.keys()):
        raw_data[row[0]] = ''
    raw_data[row[0]] += row[1]
    raw_data[row[0]] += ' '
conn.commit()

print('------------- processed articles body (news.db) -------------')

processed_raw_data = {}
proposition_list=['about', 'those','around', 'april', 'other','there', 'their','march','which','would', 'after','could','should','going','under','through','while','since']

for date in raw_data:
    keywords = raw_data[date]
    # data cleaning and parsing
    keywords = keywords.lower().replace(',', ' ').\
        replace('.', ' ').replace('/', ' ').replace('{', ' ') \
        .replace('}', ' ').replace('<', ' ').replace('>', ' ') \
        .replace('[', ' ').replace(']', ' ').replace('?', ' ') \
        .replace('(', ' ').replace(')', ' ').replace(':', ' ') \
        .replace(';', ' ').replace('*', ' ').split(' ')
    frequency_dict = {}

    for keyword in keywords:
        # filter out words with less than 5 characters (mainly propositions)
        if len(keyword) >= 5 and keyword not in set(proposition_list):
            if not keyword in set(frequency_dict.keys()):
                frequency_dict[keyword] = 0
            frequency_dict[keyword] += 1
    processed_raw_data[date] = frequency_dict

print('------------- processed keywords/frequencies (news.db) -------------')

##################################################################

# Insert entries into tables
for date in processed_raw_data:
    frequency_dict = processed_raw_data[date]
    for keyword in frequency_dict:
        c.execute('INSERT INTO keywords VALUES (?, ?, ?, ?)', (date, keyword, frequency_dict[keyword],num_articles[date]))

conn.commit()

print('------------- data inserted into table keywords (news.db) -------------')

# testing whether data had been properly stored
#c.execute('SELECT * FROM keywords')
#count = 0
#for row in c:
#    count += 1
#    print(row[0])
#    print(row[1])
#    print('---------------------')
#conn.commit()

##########################################################################################################

