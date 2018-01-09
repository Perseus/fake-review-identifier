# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:31:58 2018
"""

import numpy as np
import pandas as pd

retrieve_csv = pd.read_csv('Dataset/dataset/review.csv', delimiter='|')
training = retrieve_csv.as_matrix()

for i in range(len(training)):
    if training[i][3] >= 3:
        training[i][3] = 1
    else:
        training[i][3] = 0
train_x = [x[5] for x in training]
train_y = np.asarray([x[3] for x in training])


import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer


max_words = 10000
tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(train_x)
dictionary = tokenizer.word_index

with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

def convert_text_to_index_array(text):
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]
    
allWordIndices = []

for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)
    
allWordIndices = np.asarray(allWordIndices)

train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
train_y = keras.utils.to_categorical(train_y, 2)


from keras.models import Sequential
from keras.layers import Dense, SpatialDropout1D, LSTM, Dropout, Embedding

model = Sequential()
model.add(Embedding(len(dictionary), 32))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(32, dropout=0.2))
model.add(Dense(32, kernel_initializer='random_uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, kernel_initializer='random_uniform', activation='softmax'))


model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, train_y,
          batch_size = 32,
          epochs = 5,
          verbose = 1,
          validation_split = 0.05,
          shuffle=True)
          
