import numpy as np
import itertools
import re
import nltk
from gensim import corpora

x = "I loved the place"
corpus = []
text = re.sub('[^a-zA-Z0-9]',' ', x)
text = text.lower()
text = nltk.word_tokenize(text)
corpus.append(text)

#print(corpus)

freq_dist = nltk.FreqDist(itertools.chain(*corpus))

print(freq_dist)

vocab = freq_dist.most_common(15079)

print(vocab)

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

from keras.models import load_model
model = load_model('network_model.h5')
#model.compile(loss='categorical_crossentropy',
 #             optimizer='adam',
 #             metrics=['accuracy'])
print(model.get_weights())
predic = model.predict_classes(train_text,verbose=0)
print(predic)