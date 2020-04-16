import os
import urllib
import re
import json
from bs4 import BeautifulSoup
import requests
import sqlite3
import numpy as np
import json
import torch
import argparse
import math
import matplotlib.pyplot as plt

def preprocess():
    # Create connection to database
    conn = sqlite3.connect('news.db')
    conn2 = sqlite3.connect('news_testing.db')
    c = conn.cursor()
    c2 = conn2.cursor()

    query_severity_command = '''
    SELECT * FROM severity
    '''
    query_keyword_frequency_command = '''
    SELECT * FROM keywords
    WHERE frequency > 1
    '''
    query_num_articles_command = '''
    SELECT DISTINCT(keywords.date), num_articles FROM keywords
    '''

    article_volume_train = [] # date, num_articles
    date_map_train = {} # str(date) -> idx
    idx = 0
    c.execute(query_num_articles_command)
    for row in c:
        date = row[0]
        article_volume_train.append([date,row[1]])
        if date not in set(date_map_train.keys()):
            date_map_train[date] = idx
            idx += 1
    conn.commit()

    article_volume_test = [] # date, num_articles
    date_map_test = {} # str(date) -> idx
    idx = 0
    c2.execute(query_num_articles_command)
    for row in c2:
        date = row[0]
        article_volume_test.append([date,row[1]])
        if date not in set(date_map_test.keys()):
            date_map_test[date] = idx
            idx += 1
    conn2.commit()

    severity_train = [] # date, confirmed, death
    c.execute(query_severity_command)
    for row in c:
        date = row[0]
        confirmed = row[1]
        death = row[2]
        if date in set(date_map_train.keys()):
            severity_train.append([date, confirmed, death])
    conn.commit()
    
    severity_test = [] # date, confirmed, death
    c2.execute(query_severity_command)
    for row in c2:
        date = row[0]
        confirmed = row[1]
        death = row[2]
        if date in set(date_map_test.keys()):
            severity_test.append([date, confirmed, death])
    conn2.commit()

    keyword_occurance = {} # keyword: # occurance 
    c.execute(query_keyword_frequency_command)
    for row in c:
        keyword = row[1]
        if keyword not in set(keyword_occurance.keys()):
            keyword_occurance[keyword] = 0
        keyword_occurance[keyword] += 1
    conn.commit()

    keyword_map = {} # str(keyword) -> idx 
    idx = 0
    for keyword in list(keyword_occurance):
        if keyword_occurance[keyword] < 30:
            keyword_occurance.pop(keyword)
        else: 
            keyword_map[keyword] = idx
            idx += 1

    keyword_frequency_train = [] # date, keyword, frequency
    c.execute(query_keyword_frequency_command)
    for row in c:
        date = row[0]
        keyword = row[1]
        frequency = row[2]
        if keyword in keyword_occurance:
            keyword_frequency_train.append([date, keyword, frequency])
    conn.commit()

    keyword_frequency_test = [] # date, keyword, frequency
    c2.execute(query_keyword_frequency_command)
    for row in c2:
        date = row[0]
        keyword = row[1]
        frequency = row[2]
        if keyword in keyword_occurance:
            keyword_frequency_test.append([date, keyword, frequency])
    conn2.commit()

    num_date_train = len(article_volume_train) # 2020.1.7 - 2020.3.8
    num_date_test = len(article_volume_test) # 2020.3.28 - 2020.4.11
    num_attr_x = len(keyword_map) + 1 #  #DISTINCT(keyword) + 1 num_articles
    num_attr_y = 2 # #confirmed, #death
    news_train = torch.zeros((num_date_train, num_attr_x))
    news_test = torch.zeros((num_date_test, num_attr_x))
    cases_train = torch.zeros((num_date_train, num_attr_y))
    cases_test = torch.zeros((num_date_test, num_attr_y))

    for row in keyword_frequency_train:
        date_idx = date_map_train[row[0]]
        keyword_idx = keyword_map[row[1]]
        frequency = row[2]
        news_train[date_idx][keyword_idx] = frequency
    
    for row in keyword_frequency_test:
        date_idx = date_map_test[row[0]]
        keyword_idx = keyword_map[row[1]]
        frequency = row[2]
        if keyword_idx is not None:
            news_test[date_idx][keyword_idx] = frequency

    for row in article_volume_train:
        date_idx = date_map_train[row[0]]
        num_articles = row[1]
        news_train[date_idx][num_attr_x-1] = num_articles
    
    for row in article_volume_test:
        date_idx = date_map_test[row[0]]
        num_articles = row[1]
        news_test[date_idx][num_attr_x-1] = num_articles

    # # normalize news
    # news_sums_train = news_train.sum(axis=1)
    # news_train = news_train / news_sums_train[:, np.newaxis]
    # news_sums_test = news_test.sum(axis=1)
    # news_test = news_test / news_sums_test[:, np.newaxis]
    # # normalize cases
    # cases_sums_train = cases_train.sum(axis=1)
    # cases_train = cases_train / cases_sums_train[:, np.newaxis]
    # cases_sums_test = cases_test.sum(axis=1)
    # cases_test = cases_test / cases_sums_test[:, np.newaxis]

    for row in severity_train:
        date_idx = date_map_train[row[0]]
        cases_train[date_idx][0] = row[1] # confirmed
        cases_train[date_idx][1] = row[2] # death
    
    for row in severity_test:
        date_idx = date_map_test[row[0]]
        cases_test[date_idx][0] = row[1] # confirmed
        cases_test[date_idx][1] = row[2] # death

    # saving preprocessed data
    with open('intermediate_data/date_map_train.txt', 'w') as f:
        f.write(json.dumps(date_map_train))
    with open('intermediate_data/article_volume_train.txt', 'w') as f:
        for item in article_volume_train:
            f.write("%s\n" % item)
    with open('intermediate_data/date_map_test.txt', 'w') as f:
        f.write(json.dumps(date_map_test))
    with open('intermediate_data/article_volume_test.txt', 'w') as f:
        for item in article_volume_test:
            f.write("%s\n" % item) 
    with open('intermediate_data/severity_train.txt', 'w') as f:
        for item in severity_train:
            f.write("%s\n" % item)
    with open('intermediate_data/severity_test.txt', 'w') as f:
        for item in severity_test:
            f.write("%s\n" % item)
    with open('intermediate_data/keyword_map.txt', 'w') as f:
        f.write(json.dumps(keyword_map))
    with open('intermediate_data/keyword_frequency_train.txt', 'w') as f:
        for item in keyword_frequency_train:
            f.write("%s\n" % item)
    with open('intermediate_data/keyword_frequency_test.txt', 'w') as f:
        for item in keyword_frequency_test:
            f.write("%s\n" % item)
    torch.save(news_train, 'intermediate_data/news_train.pt')
    torch.save(cases_train, 'intermediate_data/cases_train.pt')
    torch.save(news_test, 'intermediate_data/news_test.pt')
    torch.save(cases_test, 'intermediate_data/cases_test.pt')

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def train(model, inputs, labels, loss_fn, optimizer):
    epoch_x_list = []
    epoch_y_list = []
    for epoch in range(4000):
        # shuffle
        inputs, labels = unison_shuffled_copies(inputs, labels)
        preds = model(inputs)
        loss = loss_fn(preds, labels)
        if epoch % 10 == 0:
            print('Epoch', epoch, 'Loss:', loss.item())
            epoch_x_list.append(epoch)
            epoch_y_list.append(float(loss.data))
        model.zero_grad()
        loss.backward()
        optimizer.step()
    plt.plot(epoch_x_list, epoch_y_list)
    plt.title('Logistic Regression: From News To Cases')
    # plt.title('Logistic Regression: From Cases To News')
    # plt.title('Neural Network: From News To Cases')
    # plt.title('Neural Network: From Cases To News')
    plt.xlabel('Epoch Number')
    plt.ylabel('Sigmoid Loss')
    # plt.ylabel('Mean Squared Loss')
    plt.show()

def test(model, inputs, labels, loss_fn):
    with torch.no_grad():
        preds = model(inputs)
        print(preds)
        print(labels)
        loss = loss_fn(preds, labels)
        print("Testing loss", loss.item())
        plt.plot(preds, labels)
        plt.show()

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_predict = torch.sigmoid(self.linear(x)) # to convert to 1 or 0 
        return y_predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", action="store_true",
                        help="saving inputs and labels")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-nn", "--neural_network", action="store_true",
                        help="building neural network model")
    parser.add_argument("-lr", "--logistic_regression", action="store_true",
                        help="building logistic regression model")
    parser.add_argument("-cn", "--cases_to_news", action="store_true",
                        help="using cases to predict news")
    parser.add_argument("-nc", "--news_to_cases", action="store_true",
                        help="using news to predict cases")
    args = parser.parse_args()
    
    # For preprocessing, enter: python3 analysis.py -p
    # Else, enter: python3 analysis.py -[lr]/[nn] -[cn]/[nc] -[T]/[t]
    # i.e. python3 analysis.py -lr -nc -T
    # above command runs linear regression training on news2cases

    if args.preprocess:
        print("preprocessing data...")
        preprocess()
    if args.cases_to_news:
        inputs_train = torch.load('intermediate_data/cases_train.pt')
        inputs_test = torch.load('intermediate_data/cases_test.pt')
        labels_train = torch.load('intermediate_data/news_train.pt')
        labels_test = torch.load('intermediate_data/news_test.pt')
        [_, num_attr_x] = inputs_train.size()
        [_, num_attr_y] = labels_train.size()
    if args.news_to_cases:
        inputs_train = torch.load('intermediate_data/news_train.pt')
        inputs_test = torch.load('intermediate_data/news_test.pt')
        labels_train = torch.load('intermediate_data/cases_train.pt')
        labels_test = torch.load('intermediate_data/cases_test.pt')
        [_, num_attr_x] = inputs_train.size()
        [_, num_attr_y] = labels_train.size()
    if args.neural_network:
        hidden_sz = 103
        model = torch.nn.Sequential(
            torch.nn.Linear(num_attr_x, hidden_sz),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_sz, num_attr_y)
        )
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    if args.logistic_regression:
        model = LogisticRegressionModel(num_attr_x, num_attr_y)
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), 0.01)
    if args.train:
        print("training model...")
        train(model, inputs_train, labels_train, loss_fn, optimizer)
    if args.test:
        print("testing model...")
        test(model, inputs_test, labels_test, loss_fn)