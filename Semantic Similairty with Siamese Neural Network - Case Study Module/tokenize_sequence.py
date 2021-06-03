
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

#Custom function to obtain the length of largest string in corpus :
def get_max_seq_len(corpus):
  corpus_len=[]

  for question in corpus:
    question_list=(str(question)).split()
    corpus_len.append(len(question_list))

  return max(corpus_len)



max_Seq_len=max(get_max_seq_len(data['question1']),get_max_seq_len(data['question2']))
sentence_length=max_Seq_len

#to tokenize the raw text :
tokenizer=Tokenizer(num_words=50000)
tokenizer.fit_on_texts(train_ques1+train_ques2)
word_index=tokenizer.word_index


#Converting raw text to sequences and then padding them for uniform mapping :
train_pad1 = pad_sequences(tokenizer.texts_to_sequences(train_ques1),maxlen=sentence_length)
train_pad2 = pad_sequences(tokenizer.texts_to_sequences(train_ques2),maxlen=sentence_length)


test_pad1 = pad_sequences(tokenizer.texts_to_sequences(test_ques1),maxlen=sentence_length)
test_pad2 = pad_sequences(tokenizer.texts_to_sequences(test_ques2),maxlen=sentence_length)

