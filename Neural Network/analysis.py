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
    conn = sqlite3.connect('news_testing.db')
    c = conn.cursor()

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

    article_volume = [] # date, num_articles
    date_map = {} # str(date) -> idx
    idx = 0
    c.execute(query_num_articles_command)
    for row in c:
        date = row[0]
        article_volume.append([date,row[1]])
        if date not in set(date_map.keys()):
            date_map[date] = idx
            idx += 1
    conn.commit()

    with open('date_map.txt', 'w') as f:
        f.write(json.dumps(date_map))
    with open('article_volume.txt', 'w') as f:
        for item in article_volume:
            f.write("%s\n" % item)

    severity = [] # date, confirmed, death
    c.execute(query_severity_command)
    for row in c:
        date = row[0]
        confirmed = row[1]
        death = row[2]
        if date in set(date_map.keys()):
            severity.append([date, confirmed, death])
    with open('severity.txt', 'w') as f:
        for item in severity:
            f.write("%s\n" % item)

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

    with open('keyword_map.txt', 'w') as f:
        f.write(json.dumps(keyword_map))

    keyword_frequency = [] # date, keyword, frequency
    c.execute(query_keyword_frequency_command)
    for row in c:
        date = row[0]
        keyword = row[1]
        frequency = row[2]
        if keyword in keyword_occurance:
            keyword_frequency.append([date, keyword, frequency])
    conn.commit()

    with open('keyword_frequency.txt', 'w') as f:
        for item in keyword_frequency:
            f.write("%s\n" % item)


    num_date = len(article_volume) # 2020.1.7 - 2020.3.8
    num_attr_x = len(keyword_map) + 1 #  #DISTINCT(keyword) + 1 num_articles
    num_attr_y = 2 # #confirmed, #death
    inputs = torch.zeros((num_date, num_attr_x))
    labels = torch.zeros((num_date, num_attr_y))

    for row in keyword_frequency:
        date_idx = date_map[row[0]]
        keyword_idx = keyword_map[row[1]]
        frequency = row[2]
        inputs[date_idx][keyword_idx] = frequency

    for row in article_volume:
        date_idx = date_map[row[0]]
        num_articles = row[1]
        inputs[date_idx][num_attr_x-1] = num_articles

    # normalize inputs
    row_sums = inputs.sum(axis=1)
    inputs = inputs / row_sums[:, np.newaxis]

    for row in severity:
        date_idx = date_map[row[0]]
        labels[date_idx][0] = row[1] # confirmed
        labels[date_idx][1] = row[2] # death

    torch.save(inputs, 'inputs.pt')
    torch.save(labels, 'labels.pt')

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def train(inputs, labels):
    [num_date, num_attr_x] = inputs.size()
    [_, num_attr_y] = labels.size()
    hidden_sz = 150
    model = torch.nn.Sequential(
        torch.nn.Linear(num_attr_x, hidden_sz),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_sz, num_attr_y)
    )

    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-5

    epoch_x_list = []
    epoch_y_list = []

    for epoch in range(4200):
        # shuffle
        inputs, labels = unison_shuffled_copies(inputs, labels)

        preds = model(inputs)
        loss = loss_fn(preds, labels)


        if epoch % 200 == 0:
            print("Epoch", epoch, loss.item())
            epoch_x_list.append(epoch)
            epoch_y_list.append(loss.item() * 1000)

        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    # plot data
    plt.plot(epoch_x_list, epoch_y_list)
    plt.title('Neural Network: Loss over Epoch Number (scaled by 1000)')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", action="store_true",
                        help="saving inputs and labels")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    if args.preprocess:
        print("preprocessing data...")
        preprocess()
    if args.train:
        print("training model...")
        inputs = torch.load('inputs.pt')
        labels = torch.load('labels.pt')
        train(inputs, labels)