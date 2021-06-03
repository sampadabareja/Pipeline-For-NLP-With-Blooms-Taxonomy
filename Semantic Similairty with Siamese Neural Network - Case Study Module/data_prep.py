import numpy as np
import pandas as pd
import tensorflow as tf

import nltk
import re

nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.backend as K


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM,Bidirectional,Concatenate,BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adadelta,Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU,Dense,Dropout,Lambda
from tensorflow.keras import metrics

import time

#Link to data :
data_path='/content//data/Quora/train.csv'
data=pd.read_csv(data_path)

#To Get Data Length :
data_length=len(data)

#PREPROCESS THE TEXT
import nltk 
import re 
def process_text(text):
   text = re.sub(r"it\'s","it is",str(text))
   text = re.sub(r"i\'d","i would",str(text))
   text = re.sub(r"don\'t","do not",str(text))
   text = re.sub(r"he\'s","he is",str(text)) 
   text = re.sub(r"there\'s","there is",str(text)) 
   text = re.sub(r"that\'s","that is",str(text)) 
   text = re.sub(r"can\'t", "can not", text) 
   text = re.sub(r"cannot", "can not ", text) 
   text = re.sub(r"what\'s", "what is", text) 
   text = re.sub(r"What\'s", "what is", text) 
   text = re.sub(r"\'ve ", " have ", text) 
   text = re.sub(r"n\'t", " not ", text) 
   text = re.sub(r"i\'m", "i am ", text) 
   text = re.sub(r"I\'m", "i am ", text) 
   text = re.sub(r"\'re", " are ", text) 
   text = re.sub(r"\'d", " would ", text) 
   text = re.sub(r"\'ll", " will ", text) 
   text = re.sub(r"\'s"," is",text) 
   text = re.sub(r"[0-9]"," ",str(text)) 
   text= re.sub('[^A-Za-z0-9]+'," ", text) 
   words = text.split() 
   
   return " ".join(word.lower() for word in words)



   #Manual Train Test Split :

train_size=int(data_length*0.8)
test_size=int(data_length-train_size)


#To Get the total training corpus with pre processed text :
total_train_corpus=[]
for i in range(train_size):
  
  total_train_corpus.append([ process_text(data['question1'][:train_size][i] ), process_text ( data['question2'][:train_size][i] ) ])


#To Get the total test corpus with pre processed text :
total_test_corpus=[]
for i in range(train_size,data_length):  
  total_test_corpus.append([ process_text ( data['question1'][train_size: ][i] ) , process_text ( data['question2'][train_size:][i] ) ])


#Labels -> Since they are already encoded -> 0 or 1 :

train_labels=np.array(data['is_duplicate'][:train_size])
test_labels=np.array(data['is_duplicate'][train_size:data_length])


#Manual pairing of questions :

train_ques1=[s[0] for s in total_train_corpus]
train_ques2=[s[1] for s in total_train_corpus]

test_ques1=[s[0] for s in total_test_corpus]
test_ques2=[s[1] for s in total_test_corpus]






