import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import csv



conn = sqlite3.connect('news.db')
c = conn.cursor()

get_pandemic_command = '''
SELECT date, fraction FROM keywords WHERE Keyword = "pandemic"
'''

c.execute(get_pandemic_command)
conn.commit()

fraction= []
date =[]

rows = c.fetchall()
for row in rows:
	date_row = row[0]
	date_row = date_row.replace('2020-','')
	date.append(date_row)
	fraction.append(row[1])



fig, ax = plt.subplots()
ax.bar(date, fraction)
# ax.xaxis_date()     # interpret the x-axis values as dates
plt.title('Frequency of the Keyword "pandemic" by Date')
plt.xlabel('Date')
plt.ylabel('Frequency')
fig.autofmt_xdate() # make space for and rotate the x-axis tick labels
plt.show()

#------------------------------------------------------------------------------------


get_cases_command = '''
SELECT date, fraction FROM keywords WHERE Keyword = "cases"
'''

c.execute(get_cases_command)
conn.commit()

fraction.clear()
date.clear()

rows = c.fetchall()
for row in rows:
	date_row = row[0]
	date_row = date_row.replace('2020-','')
	date.append(date_row)
	fraction.append(row[1])


fig, ax = plt.subplots()
ax.bar(date, fraction)
# ax.xaxis_date()     # interpret the x-axis values as dates
plt.title('Frequency of the Keyword "cases" by Date')
plt.xlabel('Date')
plt.ylabel('Frequency')
fig.autofmt_xdate() # make space for and rotate the x-axis tick labels
plt.show()




#------------------------------------------------------------------------------------

get_china_command = '''
SELECT date, fraction FROM keywords WHERE Keyword = "china"
'''

c.execute(get_china_command)
conn.commit()

fraction.clear()
date.clear()

rows = c.fetchall()
for row in rows:
	date_row = row[0]
	date_row = date_row.replace('2020-','')
	date.append(date_row)
	fraction.append(row[1])


fig, ax = plt.subplots()
ax.bar(date, fraction)
# ax.xaxis_date()     # interpret the x-axis values as dates
plt.title('Frequency of the Keyword "china" by Date')
plt.xlabel('Date')
plt.ylabel('Frequency')
fig.autofmt_xdate() # make space for and rotate the x-axis tick labels
plt.show()

#------------------------------------------------------------------------------------
# confirmed_data = []

# with open('time_series_covid19_confirmed_global.csv') as csvfile:
#     reader = csv.DictReader(csvfile)

#     for row in reader:
#         confirmed_data.append(row)

# confirmed_dict = confirmed_data[-2]



# date.clear()
# cases =[]


# # drop nonessential key/value pairs
# confirmed_dict.pop('Province/State')
# confirmed_dict.pop('Country/Region')
# confirmed_dict.pop('Lat')
# confirmed_dict.pop('Long')

# for k,v in confirmed_dict.items():
# 	date_fixed = k
# 	date_fixed= date_fixed.replace('/20','')
# 	date_fixed= date_fixed.replace('/19','')
# 	date.append(date_fixed)
# 	cases.append(v)


# print(date)
# print(cases)

# fig, ax = plt.subplots()
# ax.plot(date, cases)
# # ax.xaxis_date()     # interpret the x-axis values as dates
# plt.title('Number of Confirmed Cases Per Day')
# plt.xlabel('Date')
# plt.ylabel('Frequency')
# fig.autofmt_xdate() # make space for and rotate the x-axis tick labels
# plt.show()




