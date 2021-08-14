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
    df = pd.read_csv(config.datasets_dir + config.tfidf_file_name, index_col = 0)
    clean_df = pd.read_csv(config.clean_csv_name)
    df['ASSET_CLASS'] = clean_df['ASSET_CLASS']
    df['ASSET_CLASS'] = pd.Categorical(df['ASSET_CLASS'])
    df['ASSET_CLASS_CODES'] = df['ASSET_CLASS'].cat.codes
    ##########
    return df


# Description: This function splits the dataset into train and test set, n here represents - minimum number of records to be considered
def trainTestSplit(df,n):
    random.seed(123)
    df1 = df['ASSET_CLASS'].value_counts().rename_axis('Assets').reset_index(name = 'counts')
    df_new = df1[df1['counts']>=n] # Train Test split 75% - train   
    assets = list(df_new['Assets'])
    dffiltered = df[df['ASSET_CLASS'].isin(assets)]
    x = dffiltered.drop(columns = ['ASSET_CLASS','important_words','ASSET_CLASS_CODES'])
    xcols = list(x.columns)
    y = dffiltered['ASSET_CLASS_CODES']
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.3, stratify = y)
    print(' Number of Assets ' + str(len(set(list(dffiltered['ASSET_CLASS'])))))
    X_train.to_csv(config.X_train_data1, index = False)
    X_test.to_csv(config.X_test_data1, index = False)
    Y_train.to_csv(config.Y_train_data1, index = False)
    Y_test.to_csv(config.Y_test_data1, index = False)
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
    
    
# Description: Training and Saving the Naive Bayes model with the list of parameters that fits the data
def naivebayes(X_train, X_test, Y_train, Y_test):
    nb = MultinomialNB()
    print('Naive Bayes')
    nb.fit(X_train, Y_train)
    y_pred = nb.predict(X_test)
    pickle.dump(nb, open(config.nb_model_data2,"wb"))
    scores(y_pred, Y_test)
    
 # Description: Training and Saving the Decision Tree model with the list of parameters that fits the data
def decisionTree(X_train, X_test, Y_train, Y_test):
    print('Decision Tree Classifier:')
    dt = DecisionTreeClassifier(criterion = 'gini',splitter = 'best', max_features = 'log2',class_weight = None)
    dt.fit(X_train, Y_train)
    y_pred = dt.predict(X_test)
    # Saving model
    pickle.dump(dt, open(config.dt_model_data2, "wb"))
    scores(y_pred, Y_test)
    
# Description: Training and Saving the Random Forest model with the list of parameters that fits the data 
def randomForestClassifier(X_train, X_test, Y_train, Y_test):
    print('Random Forest Classifier')
    rf = RandomForestClassifier(n_estimators = 200,max_features = 'sqrt',criterion = 'gini',bootstrap = True, max_samples = 0.5)
    rf.fit(X_train, Y_train)
    y_pred = rf.predict(X_test)
    #best_5 = np.argsort(probs, axis = 1)[:,-5:]
    # Saving model
    pickle.dump(rf, open(config.rf_model_data2,"wb"))
    scores(y_pred, Y_test)

# Description: Training and Saving the KNN model with the list of parameters that fits the data
def knn(X_train, X_test, Y_train, Y_test):
    print('K Nearest Neighbors')
    clf_knn = KNeighborsClassifier(n_neighbors = 5,weights = 'distance',algorithm = 'ball_tree',metric = 'euclidean')
    clf_knn.fit(X_train, Y_train)
    y_pred = clf_knn.predict(X_test)
    pickle.dump(clf_knn, open(config.knn_model_data2, "wb"))
    scores(y_pred, Y_test)

# Description: This function calls other functions
def main():
    df = read_files()
    print(len(list(set(list(df['ASSET_CLASS'])))))
    # n = Minimum number of records
    n = 30
    x = list(df.columns)
    #for i in x:
    #    if df[i].dtypes == 'object':
    #        print(i)
    X_train, X_test, Y_train, Y_test = trainTestSplit(df,n)
    #knn(X_train, X_test, Y_train, Y_test)
    #naivebayes(X_train, X_test, Y_train, Y_test)
    #decisionTree(X_train, X_test, Y_train, Y_test)
    randomForestClassifier(X_train, X_test, Y_train, Y_test)
    #featureImportance(X_train, X_test, Y_train, Y_test)

main()
