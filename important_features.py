import pandas as pd
import config
import pickle
import numpy as np

# Read the train and test data files
def readTestFiles():
    X_train = pd.read_csv(config.X_train_data1, index_col = False)
    print(X_train.shape)
    X_test = pd.read_csv(config.X_test_data1, index_col = False)
    Y_train = pd.read_csv(config.Y_train_data1, index_col = False)
    Y_test = pd.read_csv(config.Y_test_data1, index_col = False)

    return X_train, X_test, Y_train, Y_test

# Identifies the top 20 important features generated by the randomForest Model
# This is linked to the UI for generating a plot in Important Features tab
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
    print(features)
    print(indices)
    return features, indices

# Description: This is the Main Function that invokes all other function
def main():
    X_train, X_test, Y_train, Y_test = readTestFiles()
    features, indices = importantFeatures(X_train)
    df = pd.DataFrame()
    df['features'] = features
    df['indices'] = indices
    df['indices'] = df['indices'].astype(str)
    df.to_csv(config.important_features, index = False)
    return features, indices


main()
