import numpy as np
import itertools
import re
import nltk
from gensim import corpora

x = "The place is very bad and the service was not good and the people were also very rude"

corpus = []
for i in range(len(x)):
    text = re.sub('[^a-zA-Z0-9]',' ', x[i])
    text = text.lower()
    text = nltk.word_tokenize(text)
    corpus.append(text)

#print(corpus)

freq_dist = nltk.FreqDist(itertools.chain(*corpus))

print(freq_dist)

vocab = freq_dist.most_common(15079)

print(vocab)
'''
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

from keras.preprocessing.sequence import pad_sequences
train_text = pad_sequences(train_text, maxlen=200, dtype='int32',
    padding='post', truncating='post', value=0)
print("train text", train_text)
'''
'''
from sklearn.cross_validation import ShuffleSplit
bs = ShuffleSplit(len(train_text), test_size = 0.05, random_state = 0)

for train_index, test_index in bs:
    #print("Good Morning")
    print ("TRAIN:", train_index, "TEST:", test_index)
'''    
#x_train = train_text[train_index]
#print(x_train)
'''
from keras.models import load_model
model = load_model('network_model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

predic = model.predict_classes(train_text,verbose=0)
print(predic)'''