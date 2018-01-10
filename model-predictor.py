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


predict_data = [
        "This place is horrible with very slow service. The food was not good and I didnt enjoy",
        "The food was amazing with a lovely view and ambience!",
        "The service was really good!",
        "The food was really bad",
        "The ambience was great and the service was really good too!" 
]
predict_corpus = []

for i in range(len(predict_data)):
	text = re.sub('[^a-zA-Z0-9]',' ', predict_data[i])
	text = text.lower()
	text = nltk.word_tokenize(text)
	predict_corpus.append(text)


freq_dist = nltk.FreqDist(itertools.chain(*corpus))

vocab = freq_dist.most_common(15000)

index2word = ['_'] + ['unk'] + [m[0] for m in vocab]

word2index = dict([(w,i) for i,w in enumerate(index2word)])


prediction_train_text = []
for i in range(len(predict_corpus)):
    line = []
    for word in predict_corpus[i]:
        if word in word2index:
            index = word2index[word]
        else:
            index = word2index['unk']
        line.append(index)
        print(line)
    prediction_train_text.append(line)    

prediction_train_text = pad_sequences(prediction_train_text, maxlen=200, dtype='int32',
    padding='post', truncating='post', value=0)

labels = ['negative', 'positive']
from keras.models import load_model
model = load_model('network_model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
             metrics=['accuracy'])
#print(model.get_weights())
predicter = model.predict_classes(prediction_train_text,verbose=1)
pred = model.predict(prediction_train_text)
for i in pred:
    print("%s sentiment: %f%% confidence" % (labels[np.argmax(i)], i[np.argmax(i)] * 100))
print(predicter)


'''
model.save('network_model.h5')


model.save_weights('network_weights.h5')
'''
print('######### Model Info ##############')
print(model.summary())
'''
model_json = model.to_json()
with open('network_architecture.json', 'w') as file:
    json.dump(model_json, file)
'''