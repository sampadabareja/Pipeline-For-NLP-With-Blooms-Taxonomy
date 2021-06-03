import pandas as pd
from sklearn.utils import shuffle

train_df = pd.read_csv(r'/content/gdrive/My Drive/Dataset _ QG/train.csv') # Put your directory for train.csv 
test_df = pd.read_csv(r'/content/gdrive/My Drive/Dataset _ QG/test.csv')   # Put your directory for test.csv


train_data = train_df[['context','text','question']]
test_data = test_df[['context','text','question']]


train_data = shuffle(train_data)
test_data = shuffle(test_data)


train_data.to_csv('/content/gdrive/My Drive/Dataset _ QG/train.csv',index=False)
test_data.to_csv('/content/gdrive/My Drive/Dataset _ QG/test.csv',index=False)
