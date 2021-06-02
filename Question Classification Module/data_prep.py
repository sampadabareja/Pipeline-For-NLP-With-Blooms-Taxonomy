import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv1D,LSTM,Bidirectional,Embedding,GlobalMaxPooling1D,Dropout,Flatten,MaxPool1D,MaxPooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from nltk.stem import PorterStemmer
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords




data='DEFINE THE DATA PATH HERE'
df=pd.read_csv(data)


#To get the length of max longest string
df.Text.str.len()
df.Text.str.len().max()

vocab_size=10000
embedding_dim=128
max_length=40

# Truncate and padding options
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

#Text PreProc :

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
REMOVE_NUM = re.compile('[\d+]')
STOPWORDS = set(stopwords.words('english'))
manual_stop_words = ['u','knowledge','comprehension','application','analysis','synthesis','evaluation']

def clean_text(text):
    """
    text: a string
    return: modified initial string
    """
    # lowercase text
    text = text.lower() 

    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    
    # Remove white space
    text = REMOVE_NUM.sub('', text)

    #  delete symbols which are in BAD_SYMBOLS_RE from text
    text = BAD_SYMBOLS_RE.sub('', text) 

    # delete stopwords from text
    text = ' '.join(word for word in text.split() if word not in manual_stop_words) 
    
    return text

#To Pre Process the Text : 
dataset=df
dataset['Text']=dataset['Text'].apply(clean_text)
dataset

#Converting Dataframe columns to Array values : 
text=dataset['Text'].values
labels=dataset[['Target']].values


#Performing Train Test Split to perform split of data :
X_train,y_train,X_test,y_test=train_test_split(text,labels,random_state=42,test_size=0.2)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)


