import pandas as pd
from sklearn.model_selection import train_test_split
import config
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

# Description: Reads the Tf-IDF file and converts the asset class column to numerical 
def read_files():
    tfidf_df = pd.read_csv(config.datasets_dir + config.tfidf_file_name)
    #print(list(tfidf_df.columns)[:30])
    clean_df = pd.read_csv(config.datasets_dir + config.clean_csv_name)
    df = tfidf_df
    df['ASSET_CLASS'] = clean_df['ASSET_CLASS']
    df['ASSET_CLASS'] = pd.Categorical(df['ASSET_CLASS'])
    df['ASSET_CLASS_CODES'] = df['ASSET_CLASS'].cat.codes
    return df

# Description : Generates the performance metrics for the predictions generated
def scores(y_pred, Y_test):
    print('Accuracy:   '+str(accuracy_score(y_pred, Y_test)))
    print('Precision Macro:   '+ str(precision_score(y_pred, Y_test,average = 'macro')))
    print('Recall Macro:     '+str(recall_score(y_pred, Y_test, average = 'macro')))
    print('F1 Score Macro:     '+str(f1_score(y_pred, Y_test, average = 'macro')))
    print('\n')

# Description: The dataset is split into 75% for training and 25% for testing
def trainTestSplit(df,n):
    random.seed(123)
    df1 = df['ASSET_CLASS'].value_counts().rename_axis('Assets').reset_index(name = 'counts')
    df_new = df1[df1['counts']>=n] # Train Test split 75% - train
    assets = list(df_new['Assets'])
    dffiltered = df[df['ASSET_CLASS'].isin(assets)]
    dffiltered['ASSET_CLASS_CODES'] = pd.Categorical(dffiltered['ASSET_CLASS_CODES'])
    dffiltered['ASSET_CLASS_CODES'] = dffiltered['ASSET_CLASS_CODES'].cat.codes
    x = dffiltered.drop(columns = ['ASSET_CLASS','ASSET_CLASS_CODES','important_words'])
    xcols = list(x.columns)
    y = dffiltered['ASSET_CLASS_CODES']
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.20, stratify = y)
    print(' Number of Assets ' + str(len(set(list(dffiltered['ASSET_CLASS'])))))
    #dict_codes = pd.Series(df.ASSET_CLASS.values, index = df.ASSET_CLASS_CODES).to_dict()
    return X_train, X_test, Y_train, Y_test

#Description: MLP model is trained on X train and X test and loss plots are generated 
def MLP(X_train, X_test, Y_train, Y_test):
    print(X_train.shape)
    print(X_train.shape[1])

    model = Sequential()
    model.add(Dense(500, input_dim = X_train.shape[1],activation = 'sigmoid'))
    model.add(Dense(250, activation = 'sigmoid'))
    model.add(Dense(150, activation = 'sigmoid'))
    model.add(Dense(204, activation = 'softmax'))
    # compile keras
    model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    # Early stopping
    es = EarlyStopping(monitor = 'val_loss', mode=1, verbose=3.1)
    # Fit the keras
    model.fit(X_train, Y_train, epochs = 150, batch_size = 32, validation_split = 0.1,verbose = 3.1)
    # Model Evaluation
    _, train_acc = model.evaluate(X_train, Y_train, verbose =2)
    _, test_acc = model.evaluate(X_test, Y_test, verbose=2)
    print('Train Accuracy  '+str(train_acc))
    print('Test Accuracy   '+str(test_acc))

#Description: This function invokes all other functions
def main():
    df = read_files()
    n = 100
    X_train, X_test, Y_train, Y_test = trainTestSplit(df,n)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    MLP(X_train, X_test, Y_train, Y_test)
    #print(len(Y_train.unique()))

main()
