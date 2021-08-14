'''
This program loads the various Deep Learning models from the models folder and predicts the values for the test set. It records the precision, recall, accuracy and F1-score of various models and dumps them to a csv. This CSV is later used by the UI for display.
'''

import pandas as pd
import numpy as np
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional
import config
import tensorflow as tf
from sklearn.metrics import classification_report
import pickle
import sys
from keras.models import load_model
import collections

np.random.seed(1234)
tf.random.set_seed(1234)
random.seed(1234)


def predict_with_all_dl_models():

    df = pd.DataFrame(columns=['model','accuracy','precision','recall','f1_score'])

    lstm_model = load_model(config.lstm_prepocessed_dataset1_chai)
    print("Loaded LSTM model")
    bilstm_model = load_model(config.bilstm_prepocessed_dataset1_chai)
    print("Loaded BI-LSTM model")
    rnn_model = load_model(config.rnn_prepocessed_dataset1_chai)
    print("Loaded RNN model")
    birnn_model = load_model(config.birnn_prepocessed_dataset1_chai)
    print("Loaded BI-RNN model")

    gru_model = load_model(config.gru_prepocessed_dataset1_chai)
    print("Loaded GRU model")

    bigru_model = load_model(config.bigru_prepocessed_dataset1_chai)
    print("Loaded BI-GRU model")

    models = [lstm_model,bilstm_model,rnn_model,birnn_model,gru_model,bigru_model]
    model_names = ['LSTM','Bi-LSTM','RNN','Bi-RNN','GRU','Bi-GRU']

    with open(config.X_val_DL, 'rb') as f:
        X_val = pickle.load(f)

    with open(config.y_val_DL, 'rb') as f:
        y_val = pickle.load(f)



    for index,model in enumerate(models):

        print("RUNNING PREDICTIONS FOR MODEL - ",model_names[index])

        y_pred = model.predict(X_val)
        y_pred_class = pd.DataFrame(y_pred).idxmax(axis=1)
        y_val_class = pd.DataFrame(y_val).idxmax(axis=1)

        report_dict = classification_report(y_val_class, y_pred_class,output_dict=True)

        df.loc[index,"model"] = model_names[index]
        df.loc[index,"accuracy"] = report_dict['accuracy']
        df.loc[index,"precision"] = report_dict['weighted avg']['precision']
        df.loc[index,"recall"] = report_dict['weighted avg']['recall']
        df.loc[index,"f1_score"] = report_dict['weighted avg']['f1-score']

        if(index==0):

            with open(config.code_asset_class_mapping_dict,'rb') as f:
                mapping_dict = pickle.load(f)

            mapping_dict = collections.OrderedDict(sorted(mapping_dict.items()))

            target_names_from_dict = list(mapping_dict.values())

            report_dict_with_labels = classification_report(y_val_class, y_pred_class,output_dict=True,target_names=target_names_from_dict)
            dump_df = pd.DataFrame(report_dict_with_labels).transpose()
            dump_df.to_csv(config.all_classes_performance_lstm)


    df.to_csv(config.performance_for_all_DL_models,index=False)
    return("completed execution")


if __name__=="__main__":
    predict_with_all_dl_models()


        




