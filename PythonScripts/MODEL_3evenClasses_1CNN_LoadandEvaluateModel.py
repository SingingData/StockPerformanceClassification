##############################################################################################################################################
#
# Stock future performance classification based on text
#
# Approach:
#
# Build on top of it a 1D convolutional neural network, ending in a softmax output over 3 categories.
# Use word Glove word vectors for large English text corpus as inputs model
#
# Steps
# 1) Convert all text samples in the dataset into sequences of word indices. 
# 2) A "word index" would simply be an integer ID for the word. 
# 3) We will only consider the top 100,000 most commonly occuring words in the dataset
# 4) We will truncate the sequences to a maximum length of 10,000 words.
# 5) Prepare an "embedding matrix" which will contain at index i the embedding vector for the word of index i in our word index.
# 6) Load this embedding matrix into a Keras Embedding layer, set to be frozen (its weights, the embedding vectors, will not be updated during training).
#
###############################################################################################################################################



# import libraries
from __future__ import print_function
import numpy as np
from six.moves import zip
import json
import warnings
import pandas as pd
from pandas import DataFrame   
import pickle
import re
import sys 
import azureml
import string
from scipy import stats
import pip

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer     
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers.core import Dropout
from keras.layers import LSTM
from keras.layers import Dense, Input, Flatten 
from keras.layers import Conv1D, MaxPooling1D, Embedding 
from keras.models import Model 
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.models import load_model
from keras.applications import imagenet_utils
from keras.layers import Embedding
from keras.utils.np_utils import to_categorical
import keras.backend as K
import os
import tempfile  
import logging
import gensim
from gensim.models import Phrases, phrases
from gensim.models.phrases import Phraser
from gensim.models import Word2Vec as wv
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
from IPython.display import SVG
import cloudpickle
import csv
import mkl
import matplotlib.pyplot as plt
import h5py
from keras.models import load_model
import re
import io
from os.path import dirname, join
import regex



##########################################
# Get Previously Organized Stock Data
##########################################

from azure.storage.blob import BlockBlobService
block_blob_service = BlockBlobService(account_name='nextagenda2017', account_key='aru+74LarKasMjB3BfsUM/rvPs/uGy9h7IicELf70DNWT+LoeRwbYqZIq32iLf7sSVjSSuqvKDtqr094+vUL8g==')

block_blob_service.get_blob_to_path('pklfiles', 'stockdata2.pkl','Data.pkl')
print('blobservice loaded')
 

with open('Data.pkl', 'rb') as f:
    data = pickle.load(f, encoding='utf8')
    print("Data unpickled")
data = pd.DataFrame(data)
thedata = data

np.random.seed(1337)  # for reproducibility
#################################
#Turn this into a 3 class problem
#Convert labels 1 and 3 to 2
#################################
thedata['ReturnQuantile_4Weeks'] = thedata['ReturnQuantile_4Weeks'].map({0:0,1:1,2:1,3:1,4:2})
print(thedata['ReturnQuantile_4Weeks'].unique())

#remove stop words
#from nltk.corpus import stopwords
#cachedStopWords = stopwords.words("english")

##########################################
# clean up the text in the data with regex
##########################################
thedata['fulltext'] = thedata['fulltext'].str.encode('utf-8')
thedata['fulltext'] = thedata['fulltext'].str.lower()

def clean_text(row):
    text = str(row['fulltext'])

    # Remove newline characters
    cleantext = text.replace('\r\n', ' ')

    # Convert HTML punctuation chaaracters
    cleantext = cleantext.replace('.', '')
    cleantext = cleantext.replace(',', '')
    cleantext = cleantext.replace('!', '')
    cleantext = cleantext.replace('$;', '')
    cleantext = cleantext.replace(';', '')
    cleantext = cleantext.replace(')', '')
    cleantext = cleantext.replace('(', '') 
    cleantext = cleantext.replace('', '')   
    cleantext = cleantext.replace('>', '')  
    cleantext = cleantext.replace('<', '')  
    cleantext = cleantext.replace(' the ', ' ')  
    cleantext = cleantext.replace(' is ', ' ')  
    cleantext = cleantext.replace(' of ', ' ')  
    cleantext = cleantext.replace(' then ', ' ')  
    cleantext = cleantext.replace(' their ', ' ')  
    cleantext = cleantext.replace(' this ', ' ')    
    cleantext = cleantext.replace(' in ', ' ')  
    cleantext = cleantext.replace(' and ', ' ')  
    cleantext = cleantext.replace(' our ', ' ')  
    cleantext = cleantext.replace(' by ', ' ')  
    cleantext = cleantext.replace(' to ', ' ')  
    cleantext = cleantext.replace(' at ', ' ')  
    cleantext = cleantext.replace(' on ', ' ') 
    cleantext = cleantext.replace(' for ', ' ')  
    #remove non alpha characters and specific noise
    cleantext = re.sub(r'\d+', '',cleantext)
    cleantext = re.sub(r'^b','',cleantext)
    cleantext = re.sub(r'[^\w]',' ',cleantext)
    cleantext = cleantext.replace('xc xs', '')  
    cleantext = cleantext.replace('xe xs', '')  
    cleantext = cleantext.replace('xc xS', '')  
    cleantext = cleantext.replace('xe xS', '')  
    cleantext = cleantext.replace('xc xa', '')  
    cleantext = cleantext.replace('xe xa', '')  
    cleantext = cleantext.replace(' xc xc x', '')  
    cleantext = cleantext.replace(' xc', '')  
    cleantext = cleantext.replace(' xe', '')  
    cleantext = cleantext.replace(' xs', '')  

    #remove specific noise
    cleantext = cleantext.translate(str.maketrans({'‘':' ','’':' '}))
    cleantext = cleantext.translate(str.maketrans({',':'',',':''}))
    cleantext = cleantext.translate(str.maketrans({'[':'',']':''}))
    cleantext = cleantext.translate(str.maketrans({'"':'','%':''}))
    cleantext = cleantext.translate(str.maketrans({'^':'','*':''}))

    #remove punctuation
    punctpattern = re.compile('[%s]' % re.escape(string.punctuation))
    cleanttext = re.sub(punctpattern,'', cleantext)

    #remove single letter word
    cleantext = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', cleantext) 

    # Remove extra spaces
    cleantext = re.sub('\s+', ' ', cleantext).strip()

    return cleantext
#apply regex fixes to the input text column
thedata['CleanText'] = thedata.apply(clean_text, axis=1)


#thedata['CleanText'] = thedata['CleanText'].apply(lambda x: [item for item in x if item not in cachedStopWords])

justcleandocs=thedata.drop(['fulltext'], axis=1)

justcleandocs.to_csv('cleaneddata2.tsv', sep='\t', encoding='utf-8')

################################
# Convert labels to categorical
################################

justcleandocs=thedata.drop(['fulltext','PercReturn_4_Weeks','ReturnQuantile_4Weeks'], axis=1)
justcleandocs = justcleandocs['CleanText']
#print('post regex justcleandocs',justcleandocs.head(5))

justlabels=thedata.drop(['fulltext','CleanText','PercReturn_4_Weeks'], axis=1)
justlabels=pd.DataFrame(justlabels['ReturnQuantile_4Weeks'])

#print('head of just labels',justlabels.head(5))
print(justlabels.head())
print(justlabels.tail())
print(justlabels['ReturnQuantile_4Weeks'].unique())


####################################################
#
# Set Global Vars
#
####################################################

os.chdir('C:\\')
#np.random.seed(2032)
BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove/'
MAX_SEQUENCE_LENGTH = 25000
MAX_NB_WORDS = 380000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.22


###################################################
#
# Format our text samples and labels for use in NN
#
###################################################
# Then we can format our text samples and labels into tensors that can be fed into a neural network. 
# Here we tokenize our source 'justcleandocs'
# note that the values here are ultimately indexes to the actual words

justcleandocslist  = justcleandocs.values

labels  = justlabels.values
labels_index = {}
#labels_index =  {0:0,1:1,2:2,3:3,4:4}
labels_index =  {0:0,1:1,2:2}
print('labels_index', labels_index)

#tokenize the text
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(justcleandocslist) #tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(justcleandocslist) #sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index #word_index = tokenizer.word_index
print('Found {} unique tokens'.format(len(word_index)))
print('sequences first', sequences[0])

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
sequences = None
texts = None


##################################################
#build label array from target y label in data set
##################################################
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor: ', data.shape)
print('Shape of label tensor: ', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

X_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
X_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]



################################
#Trying various optimizers
################################ 
from keras import optimizers

adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0) #keep default values
#sgd = optimizers.SGD(lr=0.003, decay=0, momentum=0.0, nesterov=True)
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#adamax = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004) #keep default values
#adagrad = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0) # keep default values

#model.compile(loss='categorical_crossentropy',
#              optimizer= rmsprop,
#              metrics=['accuracy'])

#model.fit(X_train, y_train,
#          batch_size=150,
#          epochs=3,
#          validation_data=(X_val, y_val))

          
          
from keras.models import load_model


# returns a compiled model
# identical to the previous one

model = load_model('C:\\glove\Oct4_newinit_with39.h5')       

score, acc = model.evaluate(X_val, y_val,
                            batch_size=50)

print('Test score:', score)
print('Test accuracy:', acc)


prediction = model.predict(np.array(X_val))

#https://stackoverflow.com/questions/37891954/keras-how-do-i-predict-after-i-trained-a-model
#model.predict() expects the first parameter to be a numpy array. You supply a list, which does not have the shape attribute a numpy array has.
#prediction = model.predict(np.array(tk.texts_to_sequences(text)))
print('Prediction',prediction)
print('Labels',y_val)

from tempfile import TemporaryFile
predictions = TemporaryFile()
labels = TemporaryFile()
nplabels = np.array(y_val)
np.save('C:\\glove\predictions', prediction)
np.save('C:\\glove\labels', nplabels)

#import networkx as nx
#plot_model(model, to_file='C:\\glove\Oct1_model3class_withdropout.png')

#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(model).create(prog='dot', format='svg'))


##############################
#
# More advice from Keras Site
#
##############################

#You could probably get to an even higher accuracy by training longer with some fine tuning 
#regularization method (here dropout) or by fine-tuning the Embedding layer.

#We can also test how well we would have performed by not using pre-trained word embeddings, 
#but instead initializing our Embedding layer from scratch and learning its weights during training. 
#We just need to replace our Embedding layer with the following:
#embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)

#https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
#https://arxiv.org/abs/1603.03827 
#https://nlp.stanford.edu/projects/glove/ 