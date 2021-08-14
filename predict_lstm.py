'''
This program is used to predict the asset class for the order title and line description provided as inputs. This program also performs the pre-processing required for prediction also.
'''

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import pickle
import config
import enchant
from spellchecker import SpellChecker
import tensorflow as tf
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import sys

stop_words =set(stopwords.words('english'))
english_dict = enchant.Dict("en_US")
spell = SpellChecker()




'''
#Run this part to get the embeddings_index keys, the first time.
'''
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

    with open(config.glove_embeddings_dict, 'wb') as f:
        pickle.dump(embeddings_index, f)

    with open(config.glove_embeddings_dict_keys, 'wb') as f:
        pickle.dump(list(embeddings_index.keys()), f)
    
generateEmbeddingIndex()
'''




def pre_process(order_title,line_description):
    test_string = order_title+' '+line_description
    x = re.sub('[^a-zA-Z]', ' ', test_string)
    x = x.split(' ')
    x =[word.lower() for word in x if not word in stop_words and len(word)>2]
    test_sentence = ' '.join(x)
    
    #loading the word glove dictionary - uncomment this later

    with open(config.glove_embeddings_dict_keys, 'rb') as f:
        available_embeddings_words = pickle.load(f) #dump and load these keys

    
    for index,word in enumerate(x):
        if word not in available_embeddings_words:
            res = [word[i: j] for i in range(len(word)) for j in range(i + 1, len(word) + 1) if len(word[i: j].strip())>2 and english_dict.check(word[i: j])]
            test_sentence = test_sentence.replace(word, ' '.join(res))
            
    new_vocab = list(set(test_sentence.split(' ')))
    for word in new_vocab:
        if not english_dict.check(word):
            test_sentence = test_sentence.replace(word,spell.correction(word)) #replace incorrect word with the corrected word

    return(test_sentence)




def  main(order_title,line_description):
    pre_processed_sentence = [pre_process(order_title,line_description)]
    #pre processed sentence. Return if required
    #print(pre_processed_sentence)
    
    with open(config.tokenizer_lstm, 'rb') as f:
        tokenizer_lstm = pickle.load(f)
            
    with open(config.max_length_sentence_lstm, 'rb') as f:
        max_length_sentence_lstm = pickle.load(f)
    
    with open(config.code_asset_class_mapping_dict, 'rb') as f:
        code_asset_class_mapping_dict = pickle.load(f)
    
    test_string_to_sequence = tokenizer_lstm.texts_to_sequences(pre_processed_sentence)
    test_string_final_to_predict = pad_sequences(test_string_to_sequence,maxlen=max_length_sentence_lstm)
        
    lstm_model = load_model(config.lstm_prepocessed_dataset1_chai)
    
    probs = lstm_model.predict(test_string_final_to_predict)
    top_codes = list((-probs).argsort()[:,:5])[0]
    top_preds_and_probs = {}
    
    for asset_class_code in top_codes:
        top_preds_and_probs[code_asset_class_mapping_dict[asset_class_code]] = str(probs[:,asset_class_code][0])

    
    return(top_preds_and_probs)



if __name__=="__main__":
    print(main('transformers% warehouse#123 location$ fap9989','transformers outlet made pastic case carrying transformer'))






