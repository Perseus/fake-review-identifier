# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:56:05 2018

@author: root
"""

import csv
import os

title_list = ['id', 'review', 'posneg', 'authenticity']

print(title_list)
positiveReviewPath = '/home/perseus/fake-review-identifier/Dataset/dataset/op_spam_v1.4/positive_polarity/'
negativeReviewPath = '/home/perseus/fake-review-identifier/Dataset/dataset/op_spam_v1.4/negative_polarity/'
with open('Dataset/myelott_reviews.csv', 'w') as csvfile:
    output1 = csv.DictWriter(csvfile, delimiter = '|', fieldnames = title_list)
    output1.writeheader()
    review_id = 1
    # positive authentic reviews
    for filename in os.listdir(positiveReviewPath + '/truthful_from_TripAdvisor/fold1/'):
        data = open(positiveReviewPath + '/truthful_from_TripAdvisor/fold1/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 1, 'authenticity': 1}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(positiveReviewPath + '/truthful_from_TripAdvisor/fold2/'):
        data = open(positiveReviewPath + '/truthful_from_TripAdvisor/fold2/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 1, 'authenticity': 1}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(positiveReviewPath + '/truthful_from_TripAdvisor/fold3/'):
        data = open(positiveReviewPath + '/truthful_from_TripAdvisor/fold3/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 1, 'authenticity': 1}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(positiveReviewPath + '/truthful_from_TripAdvisor/fold4/'):
        data = open(positiveReviewPath + '/truthful_from_TripAdvisor/fold4/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 1, 'authenticity': 1}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(positiveReviewPath + '/truthful_from_TripAdvisor/fold5/'):
        data = open(positiveReviewPath + '/truthful_from_TripAdvisor/fold5/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 1, 'authenticity': 1}
        output1.writerow(dataToWrite)
        review_id += 1
        
    # positive deceptive reviews
    for filename in os.listdir(positiveReviewPath + '/deceptive_from_MTurk/fold1/'):
        data = open(positiveReviewPath + '/deceptive_from_MTurk/fold1/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 1, 'authenticity': 0}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(positiveReviewPath + '/deceptive_from_MTurk/fold2/'):
        data = open(positiveReviewPath + '/deceptive_from_MTurk/fold2/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 1, 'authenticity': 0}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(positiveReviewPath + '/deceptive_from_MTurk/fold3/'):
        data = open(positiveReviewPath + '/deceptive_from_MTurk/fold3/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 1, 'authenticity': 0}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(positiveReviewPath + '/deceptive_from_MTurk/fold4/'):
        data = open(positiveReviewPath + '/deceptive_from_MTurk/fold4/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 1, 'authenticity': 0}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(positiveReviewPath + '/deceptive_from_MTurk/fold5/'):
        data = open(positiveReviewPath + '/deceptive_from_MTurk/fold5/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 1, 'authenticity': 0}
        output1.writerow(dataToWrite)
        review_id += 1
        
    # negative authentic reviews
    for filename in os.listdir(negativeReviewPath + '/truthful_from_Web/fold1/'):
        data = open(negativeReviewPath + '/truthful_from_Web/fold1/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 0, 'authenticity': 1}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(negativeReviewPath + '/truthful_from_Web/fold2/'):
        data = open(negativeReviewPath + '/truthful_from_Web/fold2/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 0, 'authenticity': 1}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(negativeReviewPath + '/truthful_from_Web/fold3/'):
        data = open(negativeReviewPath + '/truthful_from_Web/fold3/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 0, 'authenticity': 1}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(negativeReviewPath + '/truthful_from_Web/fold4/'):
        data = open(negativeReviewPath + '/truthful_from_Web/fold4/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 0, 'authenticity': 1}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(negativeReviewPath + '/truthful_from_Web/fold5/'):
        data = open(negativeReviewPath + '/truthful_from_Web/fold5/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 0, 'authenticity': 1}
        output1.writerow(dataToWrite)
        review_id += 1
    
    # negative deceptive reviews
    for filename in os.listdir(negativeReviewPath + '/deceptive_from_MTurk/fold1/'):
        data = open(negativeReviewPath + '/deceptive_from_MTurk/fold1/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 0, 'authenticity': 0}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(negativeReviewPath + '/deceptive_from_MTurk/fold2/'):
        data = open(negativeReviewPath + '/deceptive_from_MTurk/fold2/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 0, 'authenticity': 0}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(negativeReviewPath + '/deceptive_from_MTurk/fold3/'):
        data = open(negativeReviewPath + '/deceptive_from_MTurk/fold3/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 0, 'authenticity': 0}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(negativeReviewPath + '/deceptive_from_MTurk/fold4/'):
        data = open(negativeReviewPath + '/deceptive_from_MTurk/fold4/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 0, 'authenticity': 0}
        output1.writerow(dataToWrite)
        review_id += 1
    for filename in os.listdir(negativeReviewPath + '/deceptive_from_MTurk/fold5/'):
        data = open(negativeReviewPath + '/deceptive_from_MTurk/fold5/' + filename, 'r')
        dataToWrite = { 'id':review_id, 'review': data.readline(), 'posneg': 0, 'authenticity': 0}
        output1.writerow(dataToWrite)
        review_id += 1
        
    
    
    
    
    