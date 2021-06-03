
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.backend as K


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda

#Imp PARAMS :
maxsentence_length=sentence_length
embedding_dim=100
max_num_words=50000
hiddendim=128   

#Main Model:
def manhattandistance(l1,l2):
		return K.exp(-K.sum(K.abs(l1-l2), axis=1, keepdims=True))

def siamese_manhattan_network(embed_mat_weights):


	ques1 = Input(shape=(maxsentence_length,))
	ques2 = Input(shape=(maxsentence_length,))

	embedding_layer = Embedding(input_dim=max_num_words,output_dim=embedding_dim,weights=[embed_mat_weights],
    		trainable=False,input_length=maxsentence_length)

	ques1_embed = embedding_layer(ques1)
	ques2_embed = embedding_layer(ques2)

	lstm = LSTM(hiddendim,return_sequences=False)

	ques1_lstm_out = lstm(ques1_embed)
	ques2_lstm_out = lstm(ques2_embed)

	manhattan_dis = Lambda(lambda x:manhattandistance(x[0],x[1]),output_shape=lambda x:(x[0][0],1))([ques1_lstm_out,ques2_lstm_out])

	model = Model(inputs=[ques1,ques2],outputs=manhattan_dis)

	optimizer = Adam(clipnorm=1.25)

	model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

	return model

model=siamese_manhattan_network(embed_mat_weights=embedding_matrix)
model.summary()


epochs=50
batchsize=128
start = time.perf_counter()


history=model.fit([train_pad1,train_pad2],train_labels,
          validation_data=([test_pad1,test_pad2],test_labels),
          batch_size=batchsize,epochs=epochs,verbose=1)

elapsed = time.perf_counter() - start
print('Elapsed %.3f seconds.' % elapsed)

#To evaulate the model :
model.evaluate([test_pad1,test_pad2],test_labels)