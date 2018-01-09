import pandas as pd
import numpy as np
import re
import nltk
from gensim import corpora
import itertools
from keras.preprocessing.sequence import pad_sequences
import json

 
data = pd.read_csv('Dataset/dataset/review.csv', delimiter = '|')
x = data.iloc[:, 5].values
y = data.iloc[:, 3].values


    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder_y = LabelEncoder()
for i in range(0, len(y)):
    if y[i] < 3:
        y[i] = 0
    else:
        y[i] = 1
        
y = encoder_y.fit_transform(y)

y = y.reshape(-1, 1)

onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()


corpus = []
for i in range(len(x)):
    text = re.sub('[^a-zA-Z0-9]',' ', x[i])
    text = text.lower()
    text = nltk.word_tokenize(text)
    corpus.append(text)
    
#dictionary = corpora.Dictionary(corpus)
#word2index = dict(dictionary.token2id)


freq_dist = nltk.FreqDist(itertools.chain(*corpus))

vocab = freq_dist.most_common(15079)

index2word = ['_'] + ['unk'] + [m[0] for m in vocab]

word2index = dict([(w,i) for i,w in enumerate(index2word)])

train_text = []
for i in range(len(corpus)):
    line = []
    for word in corpus[i]:
        if word in word2index:
            index = word2index[word]
        else:
            index = word2index['unk']
        line.append(index)
    train_text.append(line)    

train_text = pad_sequences(train_text, maxlen=200, dtype='int32',
    padding='post', truncating='post', value=0)

from sklearn.cross_validation import ShuffleSplit
bs = ShuffleSplit(len(train_text), test_size = 0.05, random_state = 0)

for train_index, test_index in bs:
    print ("TRAIN:", train_index, "TEST:", test_index)
    
x_train = train_text[train_index]
y_train = y[train_index]
x_test = train_text[test_index]
y_test = y[test_index]

from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, SpatialDropout1D
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(len(word2index), 32, input_length=200))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(32, dropout=0.2))
model.add(Dense(32, kernel_initializer='random_uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, kernel_initializer='random_uniform', activation='softmax'))

#gru
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=50,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=32)

print('Test score:', score)
print('Test accuracy:', acc)


#prediction only
predict_data = [
        "This place is horrible with very slow service",
        "The food was amazing with a lovely view and ambience!",
        "The service was really good!",
        "The waiter was rude to us and didn't let us order in peace",
        "The ambience was great and the service was really good too!" 
]
prediction_corpus = []
for i in range(len(predict_data)):
    text = re.sub('[^a-zA-Z0-9]',' ', predict_data[i])
    text = text.lower()
    text = nltk.word_tokenize(text)
    prediction_corpus.append(text)


prediction_freq_dist = nltk.FreqDist(itertools.chain(*prediction_corpus))

prediction_vocab = prediction_freq_dist.most_common(15079)

prediction_index2word = ['_'] + ['unk'] + [m[0] for m in prediction_vocab]

prediction_word2index = dict([(w,i) for i,w in enumerate(prediction_index2word)])

prediction_train_text = []
for i in range(len(prediction_corpus)):
    line = []
    for word in prediction_corpus[i]:
        if word in prediction_word2index:
            index = prediction_word2index[word]
        else:
            index = prediction_word2index['unk']
        line.append(index)
    prediction_train_text.append(line)    

prediction_train_text = pad_sequences(prediction_train_text, maxlen=200, dtype='int32',
    padding='post', truncating='post', value=0)

predicter = model.predict_classes(prediction_train_text,verbose=0)
print(predicter)



model.save('network_model.h5')


model.save_weights('network_weights.h5')

print('######### Model Info ##############')
print(model.summary())

model_json = model.to_json()
with open('network_architecture.json', 'w') as file:
    json.dump(model_json, file)

