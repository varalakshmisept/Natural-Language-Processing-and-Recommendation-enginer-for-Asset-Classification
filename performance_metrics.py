
import pandas as pd
import config 
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np

# Description: Reads the train and test files
def readTestFiles():
    X_train = pd.read_csv(config.X_train_data1, index_col = False)
    print(X_train.shape)
    X_test = pd.read_csv(config.X_test_data1, index_col = False)
    Y_train = pd.read_csv(config.Y_train_data1, index_col = False)
    Y_test = pd.read_csv(config.Y_test_data1, index_col = False)

    return X_train, X_test, Y_train, Y_test

# Description : Generates the performance metrics for the predictions generated
def scores(y_pred, Y_test):
    print('score')
    acc = accuracy_score(y_pred, Y_test)
    pr = precision_score(y_pred, Y_test,average = 'macro')
    re = recall_score(y_pred, Y_test, average = 'macro')
    f1 = f1_score(y_pred, Y_test, average = 'macro')
    score = [acc, pr, re, f1]
    return score

# Description: This function generates a csv file with the performance metrics of the shallow learning models.
def performanceData1(X_test, Y_test):
    # Naive Bayes
    print('NB')
    nb = pickle.load(open(config.nb_model_data1,'rb'))
    y_pred = nb.predict(X_test)
    nb_scores = scores(y_pred, Y_test)
    # KNN
    print('KNN')
    knn = pickle.load(open(config.knn_model_data1, 'rb'))
    y_pred = knn.predict(X_test)
    knn_scores = scores(y_pred, Y_test)
    # DT
    print('DT')
    dt = pickle.load(open(config.dt_model_data1,'rb'))
    y_pred = dt.predict(X_test)
    dt_scores = scores(y_pred, Y_test)    
    # RF
    print('RF')
    rf = pickle.load(open(config.rf_model_data1, 'rb'))
    y_pred = rf.predict(X_test)
    rf_scores = scores(y_pred, Y_test)
    
    #creating dataframe
    df = pd.DataFrame([nb_scores, knn_scores, dt_scores, rf_scores], columns = ["accuracy", "precision", "recall", "F1-score"])
    print(df.head())
    df.to_csv(config.performance, index = False)
    return df

# Description: This function loads the Random Forest Model to generate the top 20 features that are used for predicting asset classes.

def importantFeatures(X_train):
    rf = pickle.load(open(config.rf_model_data1, 'rb'))
    names = list(X_train.columns)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    new_indices = indices[:20]
    features = X_train.columns[indices]
    indices = rf.feature_importances_[indices]
    features = list(features[:20])
    indices = list(indices[:20])
    #print(features)
    #print(indices)
    return features, indices

# Description: This function reads the deep_learning_metrics csv file to generate results dynamically on the server
def deep_learning_metrics():
    df = pd.read_csv(config.performance_for_all_DL_models)
    return list(df['accuracy']), list(df['precision']), list(df['recall']), list(df['f1_score'])

# Description: This function invokes all other functions
def main():
    X_train, X_test, Y_train, Y_test = readTestFiles()
    df = performanceData1(X_test, Y_test)
    df = pd.read_csv(config.performance)
    #print(list(df['accuracy']))
    return list(df['accuracy']), list(df['precision']), list(df['recall']), list(df['F1-score'])
    #features, indices = importantFeatures(X_train)



main()
