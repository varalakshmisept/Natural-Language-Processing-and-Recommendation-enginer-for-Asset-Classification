# All the models are trained on dataset 3
import pandas as pd
import config
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# Description: Reads the Tf-IDF file and converts the asset class column to numerical 
def read_files():
    df = pd.read_csv(config.datasets_dir + config.optimized_dataset)
    df['ASSET_CLASS'] = pd.Categorical(df['ASSET_CLASS'])
    df['ASSET_CLASS_CODES'] = df['ASSET_CLASS'].cat.codes
    ##########
    x = df['ASSET_CLASS'].value_counts().rename_axis('Assets').reset_index(name = 'counts')
    x = x[x['counts']<100]
    print(x['Assets'])
    print(x)
    return df

# Description: This function splits the dataset into train and test set, n here represents - minimum number of records to be considered
def trainTestSplit(df,n):
    random.seed(123)
    df1 = df['ASSET_CLASS'].value_counts().rename_axis('Assets').reset_index(name = 'counts')
    df_new = df1[df1['counts']>=n] # Train Test split 75% - train   
    assets = list(df_new['Assets'])
    dffiltered = df[df['ASSET_CLASS'].isin(assets)]
    x = dffiltered.drop(columns = ['ASSET_CLASS','important_words','BUSINESS_UNIT','PSC_CODE','FUND_SUBOBJCLASS','ORDER_DATE','ORDER_TITLE','ASSET_CLASS_CODES',                                                                                                                                                           
            'LINE_DESCRIPTION', 'VENDOR_NAME', 'VENDOR_COUNTRY', 'ASSET_CLASS_DESCRIPTION','text_fields','ASSET_CLASS_OLD','SUB_OBJ_DESCR','OBJ_CODE'])
    xcols = list(x.columns)
    y = dffiltered['ASSET_CLASS_CODES']
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.20, stratify = y)
    print(' Number of Assets ' + str(len(set(list(dffiltered['ASSET_CLASS'])))))
    print(X_train.shape)
    return X_train, X_test, Y_train, Y_test

# Description : Generates the performance metrics for the predictions generated
def scores(y_pred, Y_test):
    print('Accuracy:   '+str(accuracy_score(y_pred, Y_test)))
    print('Precision Macro:   '+ str(precision_score(y_pred, Y_test,average = 'macro')))
    print('Recall Macro:     '+str(recall_score(y_pred, Y_test, average = 'macro')))
    print('F1 Score Macro:     '+str(f1_score(y_pred, Y_test, average = 'macro')))
    print('\n')


# Description: Creates a csv file on test set with the predicted class probabilities
def createDataFrame(Y_test, y_pred, y_pred_proba,name):
    df = y_pred_proba
    df['Y_test'] = Y_test
    df['y_pred'] = y_pred
    df.to_csv(name)

# Description: Classification report is generated on the models trained
def report (y_pred, Y_test, labels):
    print(classification_report(Y_test, y_pred))

# Description: Training and Saving the Naive Bayes model with the list of parameters that fits the data3
def naivebayes(X_train, X_test, Y_train, Y_test):
    nb = MultinomialNB()
    print('Naive Bayes')
    nb.fit(X_train, Y_train)
    y_pred = nb.predict(X_test)
    probs = pd.DataFrame(nb.predict_proba(X_test))
    createDataFrame(Y_test, y_pred, probs, config.nb_results_data3)
    pickle.dump(nb, open(config.nb_model_data3,"wb"))
    scores(y_pred, Y_test)

 # Description: Training and Saving the Decision Tree model with the list of parameters that fits the data3
def decisionTree(X_train, X_test, Y_train, Y_test):
    print('Decision Tree Classifier:')
    dt = DecisionTreeClassifier(criterion = 'entropy',splitter = 'best', max_features = 'auto',class_weight = None)
    dt.fit(X_train, Y_train)
    y_pred = dt.predict(X_test)
    # Saving model
    probs = pd.DataFrame(dt.predict_proba(X_test))
    createDataFrame(Y_test, y_pred, probs, config.dt_results_data3)
    pickle.dump(dt, open(config.dt_model_data3, "wb"))
    scores(y_pred, Y_test)
   
 # Description: Training and Saving the Random Forest model with the list of parameters that fits the data 3
def randomForestClassifier(X_train, X_test, Y_train, Y_test):
    print('Random Forest Classifier')
    rf = RandomForestClassifier(n_estimators = 500,max_features = 'auto',criterion = 'gini',bootstrap = True, max_samples = 0.75, oob_score = False, warm_start = False)
    rf.fit(X_train, Y_train)
    y_pred = rf.predict(X_test)
    probs = pd.DataFrame(rf.predict_proba(X_test))
    #best_5 = np.argsort(probs, axis = 1)[:,-5:]
    createDataFrame(Y_test, y_pred, probs, config.rf_results_data3)
    # Saving model
    pickle.dump(rf, open(config.rf_model_data3,"wb"))
    scores(y_pred, Y_test)

# Description: Training and Saving the KNN model with the list of parameters that fits the data3
def knn(X_train, X_test, Y_train, Y_test):
    print('K Nearest Neighbors')
    clf_knn = KNeighborsClassifier(n_neighbors = 5,weights = 'distance',algorithm = 'ball_tree',metric = 'euclidean')
    clf_knn.fit(X_train, Y_train)
    y_pred = clf_knn.predict(X_test)
    probs = pd.DataFrame(clf_knn.predict_proba(X_test))
    createDataFrame(Y_test, y_pred, probs, config.knn_results_data3)
    pickle.dump(clf_knn, open(config.knn_model_data3, "wb"))
    scores(y_pred, Y_test)

# Description: Top 20 Important Features that are used in predicting asset classes
def featureImportance(X_train, X_test, Y_train, Y_test):
    rf = pickle.load(open(config.rf_model_data3, 'rb'))
    names = list(X_train.columns)
    y = sorted(zip(map(lambda x:round(x,4), rf.feature_importances_), names),reverse = True)
    print('Feature Importances')
    #importances = rf.feature_importances_ 
    #indices = np.argsort(importances)
    for i in y[:20]:
        col = i[1]
        train = X_train[col]
        test = X_test[col]
        rf = RandomForestClassifier()
        rf.fit(train,Y_train)
        y_pred = rf.predict(test) 
        print(col+'   '+accuracy_score(y_pred,Y_test))


# Description: This function calls other functions
def main():
    df = read_files()
    print(len(list(set(list(df['ASSET_CLASS'])))))
    # n = Minimum number of records
    n = 5
    x = list(df.columns)
    #for i in x:
    #    if df[i].dtypes == 'object':
    #        print(i)
    X_train, X_test, Y_train, Y_test = trainTestSplit(df,n)
    knn(X_train, X_test, Y_train, Y_test)
    naivebayes(X_train, X_test, Y_train, Y_test)
    decisionTree(X_train, X_test, Y_train, Y_test)
    randomForestClassifier(X_train, X_test, Y_train, Y_test)
    #featureImportance(X_train, X_test, Y_train, Y_test)

main()
