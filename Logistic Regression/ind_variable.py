import csv
import os
import urllib
import re
import json
from bs4 import BeautifulSoup
import requests
import sqlite3

confirmed_data = []
death_data = []

##################################################################

with open('time_series_covid19_confirmed_global.csv') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        confirmed_data.append(row)

with open('time_series_covid19_deaths_global.csv') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        death_data.append(row)

##################################################################

# map each date to the total number of confirmed cases in the world
confirmed_dict = confirmed_data[-1]
# map each date to the total number of deaths in the world
death_dict = death_data[-1]

# drop nonessential key/value pairs
confirmed_dict.pop('Province/State')
confirmed_dict.pop('Country/Region')
confirmed_dict.pop('Lat')
confirmed_dict.pop('Long')
death_dict.pop('Province/State')
death_dict.pop('Country/Region')
death_dict.pop('Lat')
death_dict.pop('Long')

##################################################################

# Create connection to database
conn = sqlite3.connect('news.db')
c = conn.cursor()

# Delete tables if they exist
c.execute('DROP TABLE IF EXISTS "severity";')

# Create tables in the database and add data to it.
create_severity_table_command = '''
CREATE TABLE IF NOT EXISTS severity (
    date VARCHAR(255) NOT NULL,
    confirmed_no int NOT NULL,
    deaths_no int NOT NULL
	);
'''

c.execute(create_severity_table_command)
conn.commit()

##################################################################

# Insert entries into tables
for date in confirmed_dict:
    confirmed_number = confirmed_dict[date]
    deaths_number = death_dict[date]
    month, day, year = date.split('/')
    if len(day) == 1:
        day = '0'+day
    if len(month) == 1:
        month = '0'+month
    formatted_date = '20'+year+'-'+month+'-'+day
    c.execute('INSERT INTO severity VALUES (?, ?, ?)', (formatted_date, confirmed_number, deaths_number))

conn.commit()

c.execute('SELECT * FROM severity')

# test whether severity table is correctly implemented and data is stored
#for row in c:
#   print(row)