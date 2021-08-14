#!/usr/bin/env python
# coding: utf-8

'''
Program Description: This program builds a Bi-GRU model over the GloVe embeddings on the words obtained after performing the pre-processing over the concatenated string - Order Title and Line Description
'''

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional, SimpleRNN, GRU
import config
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report

np.random.seed(1234)
tf.random.set_seed(1234)
random.seed(1234)


# In[2]:

'''
Description: This function encodes the asset class to sequential categorical values and splits the train and test sets to 80% and 20% respectively.
'''

def trainTestSplit(df,n):
    
    df1 = df['ASSET_CLASS'].value_counts().rename_axis('Assets').reset_index(name = 'counts')
    df_new = df1[df1['counts']>=n] # Train Test split 75% - train
    assets = list(df_new['Assets'])
    dffiltered = df[df['ASSET_CLASS'].isin(assets)]
    dffiltered['ASSET_CLASS_CODES'] = pd.Categorical(dffiltered['ASSET_CLASS'])
    dffiltered['ASSET_CLASS_CODES'] = dffiltered['ASSET_CLASS_CODES'].cat.codes
    
    x = dffiltered['SPELL_CORRECTED']
    y = pd.get_dummies(dffiltered['ASSET_CLASS_CODES']) 
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.20, stratify = y)
    print(' Number of Assets ' + str(len(set(list(dffiltered['ASSET_CLASS'])))))
    return X_train, X_test,  Y_train, Y_test



'''
Description: Generates the dictionary of GloVe embeddings from the GloVe object.
'''

def generateEmbeddingIndex():
    print('Indexing word vectors.')
    embeddings_index = {}
    with open((config.utils_dir+config.glove_txt_300d)) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


'''
Description: This function does the pre-processing and the training associated with the Bi-GRU model over the train and test sets.
'''


def bilstm(X_train, X_test, Y_train, Y_test,wordembeddings):
    np.random.seed(1234)
    tf.random.set_seed(1234)
    random.seed(1234)
    
    max_length_sentence = X_train.str.split().str.len().max()
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True)
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index
    EMBEDDING_DIM=300
    vocabulary_size=len(word_index)+1
    print('Found %s unique tokens.' % len(word_index))
    
    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_valid=tokenizer.texts_to_sequences(X_test)
    X_train = pad_sequences(sequences_train,maxlen=max_length_sentence)
    X_val = pad_sequences(sequences_valid,maxlen=X_train.shape[1])
    y_train = np.asarray(Y_train)
    y_val = np.asarray(Y_test)
    #print(word_index)
    
    '''
    print('Shape of data tensor:', X_train.shape)
    print('Shape of data tensor:', X_val.shape)
    print('Shape of data tensor:', y_train.shape)
    print('Shape of data tensor:', y_val.shape)
    
    print(X_train)
    print("*"*100)
    print(X_val)
    print("*"*100)
    print(y_train)
    print("*"*100)
    print(y_val)
    '''
    
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        if(word in wordembeddings.keys()):
            embedding_vector = wordembeddings[word]
            if len(embedding_vector)==0: #if array is empty
                embedding_vector = wordembeddings[word.title()]
                if len(embedding_vector)==0:
                    embedding_vector = wordembeddings[word.upper()]
                    if len(embedding_vector)==0:
                        embedding_vector = np.array([round(np.random.rand(),8) for i in range(0,300)])
                        
        else:
            #print("WORD NOT IN DICT",word)
            embedding_vector = np.array([round(np.random.rand(),8) for i in range(0,300)])
            
        if len(embedding_vector)!=0:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=False) #Try with True
    
    
    inputs = Input(shape=(X_train.shape[1],))
    model = (Embedding(vocabulary_size, EMBEDDING_DIM, input_length=max_length_sentence,weights=[embedding_matrix]))(inputs)
    
    model = Bidirectional(GRU(64))(model) # !!!!!!! CHANGE THIS FOR OTHER MODELS
    model = (Dense(900, activation='relu'))(model)
    model = (Dense(400, activation='relu'))(model)
    model = (Dense(250, activation='relu'))(model)
    model = (Dense(204, activation='softmax'))(model)
    model = Model(inputs=inputs,outputs=model)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    callbacks = [EarlyStopping(monitor='val_loss')]
    hist_adam = model.fit(X_train, y_train, batch_size=1000, epochs=200, verbose=1, validation_data=(X_val, y_val),callbacks=callbacks)     #!!!!!!!!!!!!!!!!!!!!!!!CHANGE BATCH SIZE TO 1000 #change epochs to 200
    
    model.save(config.bigru_prepocessed_dataset1_chai) # !!!!!!! CHANGE THIS FOR OTHER MODELS
    
    y_pred = model.predict(X_val)
    print(y_pred)
    
    y_val_class = pd.DataFrame(y_val).idxmax(axis=1)
    print(y_val_class)
    
    y_val_class_argmax = np.argmax(y_val,axis=1)
    y_pred_class_argmax = np.argmax(y_pred,axis=1)
    
    y_pred_class = pd.DataFrame(y_pred).idxmax(axis=1)
    print(y_pred_class)
    
    
    print(classification_report(y_val_class, y_pred_class))
    
    plt.suptitle('Optimizer : Adam', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.plot(hist_adam.history['loss'], color='b', label='Training Loss')
    plt.plot(hist_adam.history['val_loss'], color='r', label='Validation Loss')
    plt.legend(loc='upper right')
    
    plt.savefig('/home/ubuntu/asset_classification/results/bigru_model_dataset1_preprocessed_chai.png') # !!!!!!! CHANGE THIS FOR OTHER MODELS
    
    tf.keras.utils.plot_model(model, to_file=config.bigru_architecture, show_shapes=True) # !!!!!!! CHANGE THIS FOR OTHER MODELS
    
    return(y_pred,y_val_class,y_pred_class,y_val_class_argmax,y_pred_class_argmax)
    

'''
Description: This is the main function that invokes all the other functions.
'''

def main_train():
    df = pd.read_csv(config.datasets_dir+config.final_preprocessed)
    df.head()

    df = df.replace(np.nan, '', regex = True)
    df["SPELL_CORRECTED"].isnull().values.any()

    X_train, X_test, Y_train, Y_test = trainTestSplit(df,100)

    print(X_train.head())

    print(X_test.head())

    print(Y_train.head())

    print(Y_test.head())


    print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


    wordembeddings = generateEmbeddingIndex()

    y_pred,y_val_class,y_pred_class,y_val_class_argmax,y_pred_class_argmax = bilstm(X_train, X_test, Y_train, Y_test, wordembeddings)

    print("Execution done")

if __name__=="__main__":
    main_train()

