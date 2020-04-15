#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains the main program to read data, run classifiers
   and print results to stdout.
   You do not need to change this file. You can add debugging code or code to
   help produce your report, but this code should not be run by default in
   your final submission.
   Brown CS142, Spring 2020
"""

import numpy as np
import pandas as pd
from models import NaiveBayes
import sqlite3

def get_credit():
    """
    Gets and preprocesses German Credit data
    """
    data = pd.read_csv('./data/german_numerical-binsensitive.csv') # Reads file - may change

    # MONTH categorizing
    data['month'] = pd.cut(data['month'],3, labels=['month_1', 'month_2', 'month_3'], retbins=True)[0]
    # month bins: [ 3.932     , 26.66666667, 49.33333333, 72.        ]
    a = pd.get_dummies(data['month'])
    data = pd.concat([data, a], axis = 1)
    data = data.drop(['month'], axis=1)

    # CREDIT categorizing
    data['credit_amount'] = pd.cut(data['credit_amount'], 3, labels=['cred_amt_1', 'cred_amt_2', 'cred_amt_3'], retbins=True)[0]
    # credit bins: [  231.826,  6308.   , 12366.   , 18424.   ]
    a = pd.get_dummies(data['credit_amount'])
    data = pd.concat([data, a], axis = 1)
    data = data.drop(['credit_amount'], axis=1)

    for header in ['investment_as_income_percentage', 'residence_since', 'number_of_credits']:
        a = pd.get_dummies(data[header], prefix=header)
        data = pd.concat([data, a], axis = 1)
        data = data.drop([header], axis=1)

    # change from 1-2 classes to 0-1 classes
    data['people_liable_for'] = data['people_liable_for'] -1
    data['credit'] = -1*(data['credit']) + 2 # original encoding 1: good, 2: bad. we switch to 1: good, 0: bad

    # balance dataset
    data = data.reindex(np.random.permutation(data.index)) # shuffle
    pos = data.loc[data['credit'] == 1]
    neg = data.loc[data['credit'] == 0][:350]
    combined = pd.concat([pos, neg])

    y = data.iloc[:, data.columns == 'credit'].to_numpy()
    x = data.drop(['credit', 'sex', 'age', 'sex-age'], axis=1).to_numpy()

    # split into train and validation
    X_train, X_val, y_train, y_val = x[:350, :], x[351:526, :], y[:350, :].reshape([350,]), y[351:526, :].reshape([175,])

    # keep info about sex and age of validation rows for fairness portion
    x_sex = data.iloc[:, data.columns == 'sex'].to_numpy()[351:526].reshape([175,])
    x_age = data.iloc[:, data.columns == 'age'].to_numpy()[351:526].reshape([175,])
    x_sex_age = data.iloc[:, data.columns == 'sex-age'].to_numpy()[351:526].reshape([175,])

    return X_train, X_val, y_train, y_val, x_sex, x_age, x_sex_age

def get_news():
    '''
    conn = sqlite3.connect('news.db')
    c = conn.cursor()
    query_all_words_command = '''
    #SELECT keyword FROM keywords group by KEYWORD having COUNT(KEYWORD)>50
    '''
    c.execute(query_all_words_command)
    all_words = set()
    for row in c:
        word = row[0].strip('():"').strip()
        if word not in all_words:
            all_words.add(word)
    all_words = list(all_words)
    col_num = len(all_words)
    print('counted all words')

    query_date_command = '''
    #SELECT date FROM keywords group by date
    '''
    c.execute(query_date_command)
    row_num = 0
    for row in c:
        row_num += 1
    X_train = np.zeros((row_num, col_num))

    query_date_command = '''
    #SELECT date, keyword FROM keywords WHERE keyword IN (SELECT keyword FROM keywords group by KEYWORD having COUNT(KEYWORD)>50)
    '''
    c.execute(query_date_command)
    i = 0
    curr_date = '2020-01-07'
    for row in c:
        if curr_date != row[0]:
            #print(X_train)
            i += 1
            curr_date = row[0]
        print(row[0])
        X_train[i, all_words.index(row[1].strip('():"').strip())] = 1

    np.save('X_train.npy', X_train)
    '''
    #--------------------------

    '''
    conn = sqlite3.connect('news.db')
    c = conn.cursor()
    query_all_words_command = '''
    #SELECT keyword FROM keywords group by KEYWORD having COUNT(KEYWORD)>50
    '''
    c.execute(query_all_words_command)
    all_words = set()
    for row in c:
        word = row[0].strip('():"').strip()
        if word not in all_words:
            all_words.add(word)
    all_words = list(all_words)
    col_num = len(all_words)
    print('counted all words')

    conn = sqlite3.connect('news_testing.db')
    c = conn.cursor()
    query_date_command = '''
    #SELECT date FROM keywords group by date
    '''
    c.execute(query_date_command)
    row_num = 0
    for row in c:
        row_num += 1
    X_val = np.zeros((row_num, col_num))

    query_date_command = '''
    #SELECT date, keyword FROM keywords WHERE keyword IN (SELECT keyword FROM keywords group by KEYWORD having COUNT(KEYWORD)>2)
    '''
    c.execute(query_date_command)
    i = 0
    curr_date = '2020-03-28'
    for row in c:
        print(curr_date)
        if curr_date != row[0]:
            #print(X_train)
            i += 1
            curr_date = row[0]
        print(row[0])
        try:
            X_val[i, all_words.index(row[1].strip('():"').strip())] = 1
        except:
            continue
    print(X_val.nonzero())

    np.save('X_val.npy', X_val)
    '''

    X_train = np.load('X_train.npy')

    X_val = np.load('X_val.npy')

    print(len(X_val.nonzero()[0]))
    growth = [0,0,0,0,0,0,0,0,0,0,0,0.088888889,0.274193548,0.686868687,0.23255814,0.535135135,0.151376147,0.304994687,0.343793584,0.322946176,0.27639221,0.47525995,0.095361661,0.251153753,0.170544978,0.175361356,0.28289748,0.155625975,0.167880462,0.135444183,0.102584919,0.104591317,0.073518319,0.075466999,0.061082269,0.045533682,0.009265607,0.250911079,0.097435897,0.031073446,0.030804223,0.027764886,0.024994676,0.006650009,0.007323123,0.008096955,0.022310747,0.004888675,0.00757909,0.010509166,0.012065661,0.016411669,0.016240251,0.022078571,0.026683566,0.021449295,0.02729427,0.023969722,0.028257361,0.038457383,0.038224985,0.036186157]
    growth_val = [0.102017124,0.082549227,0.079562724,0.08757917,0.080546426,0.079786594,0.075234712,0.084758913,0.058726609,0.054260609,0.056794914,0.056255559,0.052807221,0.056965134,0.045043392]

    y_val = np.array(growth_val) > 0.1
    y_train = np.array(growth) > 0.1
    #print(y_val)
    return X_train, X_val, y_train, y_val

def main():

    np.random.seed(0)

    #X_train, X_val, y_train, y_val, x_sex, x_age, x_sex_age = get_news()
    X_train, X_val, y_train, y_val = get_news()

    model = NaiveBayes(2)

    model.train(X_train, y_train)

    print("------------------------------------------------------------")

    print("Train accuracy:")
    print(model.accuracy(X_train, y_train))

    print("------------------------------------------------------------")

    print("Test accuracy:")
    print(model.accuracy(X_val, y_val))

    print("------------------------------------------------------------")
    '''
    print("Fairness measures:")
    model.print_fairness(X_val, y_val, x_sex_age)
    '''

if __name__ == "__main__":
    main()
