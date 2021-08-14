'''
This script is used for pre-processing the dataset in order to facilitate Deep Learning models.
'''


import config as config
import csv
import pandas as pd
import boto3
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
import sys
import enchant
import json
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
try:
    from nltk.corpus import words
except:
    nltk.download('words')
from spellchecker import SpellChecker
spell = SpellChecker()
english_dict = enchant.Dict("en_US")



stop_words =set(stopwords.words('english'))

def connect():
    bucket = config.bucket
    data_key = config.data_key
    data_location = 's3://{}/{}'.format(bucket, data_key)
    return data_location

def pre_process(x):
    x=re.sub('[^a-zA-Z]',' ',x)
    x=x.split()
    for index,word in enumerate(x):
        if word in config.replacement_dict.keys():
            x[index]=config.replacement_dict[word]
    
    x =[word.lower() for word in x if not word in stop_words and len(word)>2]
    x = ' '.join(x)
    x = re.sub('[^a-zA-Z]', ' ', x)
    x = re.sub(' +',' ',x)
    return(x)


def get_missing_word_embedding_words(df):
    #f = open("words_not_in_corpus",'w')
    corpus_text = ' '.join(df[:]['text_fields'])
    vocabulary = list(set(corpus_text.split(' ')))
    print("Length of Vocabulary",len(vocabulary))

    glove_input = config.utils_dir + config.glove_txt_300d
    output_file = config.utils_dir+"vectors.txt"
    glove2word2vec(glove_input,output_file)
    print("Model Loading ...")
    model = KeyedVectors.load_word2vec_format(output_file, binary=False)
    print("Model Loaded.")
    missing_embeddings_words = []
    for word in range(len(vocabulary)):
        try:
            model[vocabulary[word]]
        except:
            missing_embeddings_words.append(vocabulary[word])
            #f.write('\n')

    print("Number of words that do not have wrod embeddings: ",len(missing_embeddings_words))
    return(missing_embeddings_words) 


def get_substrings(missing_embeddings_words):
    
    #words_list = words.words()
    word_substrings_dict = {}
    temp_list = []
    for test_str in missing_embeddings_words:
        temp_list = [test_str[i: j] for i in range(len(test_str)) for j in range(i + 1, len(test_str) + 1) if len(test_str[i: j])>2 and english_dict.check(test_str[i: j])]
        
        if(len(temp_list)!=0):
            word_substrings_dict[test_str] = ' '.join(temp_list)
    return(word_substrings_dict)

def correctSpellings(new_vocab):
    replace_spellings = {}
    i = 0
    print(i)
    fname = config.datasets_dir + config.correctspelling
    f = open(fname,"w")
    for word in new_vocab:
        print("In THE LOOP....", word)
        print(i)
        i = i+1
        if(word):
            if not english_dict.check(word):
                replace_spellings[word] = spell.correction(word)
                f.write(word+':'+spell.correction(word))
                f.write("\n")
                print("word replaced")

    # dump to file
    f.close()

    return replace_spellings

def replaceFunction(d, text,regex):
     # Create a regular expression  from the dictionary keys
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: d[mo.string[mo.start():mo.end()]], text)

def find_substrings_check_dictionary(df, missing_embeddings_words):
    word_substrings_dict = get_substrings(missing_embeddings_words)
    #print(word_substrings_dict)

    #replacing the words that do not have embeddings with their substrings
    #df["PROCESSED_TEXT_FIELDS"] = df["text_fields"].replace(word_substrings_dict,regex=True)
    regex = re.compile("(%s)" % "|".join(["\\b" + x + "\\b" for x in word_substrings_dict.keys()]))
    df["PROCESSED_TEXT_FIELDS"] = df["text_fields"].apply(lambda x: replaceFunction(word_substrings_dict, x,regex))
   
    #print("NEW VOCAB ..")
    new_vocab = list(set(' '.join(df["PROCESSED_TEXT_FIELDS"]).split(" ")))

    replace_spellings = {}
    
    print('SPELL CORRECTION ..')    
    
    replacespellings = correctSpellings(new_vocab)

        #print(replace_spellings)

    # dump to file
    #df['PROCESSED_TEXT_FIELDS_SPELL'] = df["PROCESSED_TEXT_FIELDS"].replace(replace_spellings,regex=True)

    return(df)

def replaceCorrectSpellings(df):
    fname = config.datasets_dir + config.correctspelling
    f = open(fname,"r")
    replaceSpelling = {}
    lines = f.readlines()
    j = 0
    for i in lines:
        print("In THE LOOP....")
        print(j)
        j = j+1
        x = i.split(':')
        replaceSpelling[x[0]] = x[1]

    df['PROCESSED_TEXT_FIELDS_SPELL'] = df["PROCESSED_TEXT_FIELDS"].replace(replaceSpelling, regex = True)
    return df
                        


def main():
    data_location = connect()
    df = pd.read_csv(data_location)
    df = df.replace(np.nan, '', regex = True)
    df['ORDER_TITLE'] = df['ORDER_TITLE'].apply(pre_process)
    df["LINE_DESCRIPTION"] = df["LINE_DESCRIPTION"].apply(pre_process)
    df["text_fields"] = df['ORDER_TITLE']+' '+df["LINE_DESCRIPTION"]

    print("missing embeddings words start")
    missing_embeddings_words = get_missing_word_embedding_words(df)
    print("missing embeddings words end")

    #missing_embeddings_words = ['transformers','cyberlink','nnjfqi','masrm','giratoria']

    df = find_substrings_check_dictionary(df,missing_embeddings_words)

    #df = replaceCorrectSpellings(df)
    print('successfully replace')
    df.to_csv(config.datasets_dir+config.final_preprocessed,index=False)
    print('Saved df')

if __name__ == "__main__":
    main()
