# All the models are trained on dataset 1 and dataset2
import pandas as pd
import config
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
import random
from sklearn.metrics import classification_report

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

# Description: This function splits the dataset into train and test set, n here represents - minimum number of records to be considered
def trainTestSplit(df,n):
    random.seed(123)
    df1 = df['ASSET_CLASS'].value_counts().rename_axis('Assets').reset_index(name = 'counts')
    df_new = df1[df1['counts']>=n] # Train Test split 75% - train   
    assets = list(df_new['Assets'])
    dffiltered = df[df['ASSET_CLASS'].isin(assets)]
    x = dffiltered.drop(columns = ['ASSET_CLASS','ASSET_CLASS_CODES','important_words'])
    xcols = list(x.columns)
    y = dffiltered['ASSET_CLASS_CODES']
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.3, stratify = y)
    print(' Number of Assets ' + str(len(set(list(dffiltered['ASSET_CLASS'])))))
    #dict_codes = pd.Series(df.ASSET_CLASS.values, index = df.ASSET_CLASS_CODES).to_dict()
    return X_train, X_test, Y_train, Y_test

# Description : Generates the performance metrics for the predictions generated
def scores(y_pred, Y_test):
    print('Hiiii')
    print('Accuracy:   '+str(accuracy_score(y_pred, Y_test)))
    print('Precision Macro:   '+ str(precision_score(y_pred, Y_test,average = 'macro')))
    print('Recall Macro:     '+str(recall_score(y_pred, Y_test, average = 'macro')))
    print('F1 Score Macro:     '+str(f1_score(y_pred, Y_test, average = 'macro')))
    print('\n')

# Description: Classification report is generated on the models trained
def report (y_pred, Y_test, labels):
    print(classification_report(Y_test, y_pred))

# Description: Naive Bayes model is trained on X train and Y train
def naivebayes(X_train, X_test, Y_train, Y_test):
    nb = MultinomialNB()
    print('Naive Bayes')
    nb.fit(X_train, Y_train)
    y_pred = nb.predict(X_test)
    scores(y_pred, Y_test)

# Description: Grid Search is performed on train set using Decision tree model to identify the best set of parameters that fits the data
def decisionTree(X_train, X_test, Y_train, Y_test):
    dt = DecisionTreeClassifier()
    grid_values = {'criterion':['gini','entropy'],
                    'splitter':['best','random'],
                    'max_features':['auto','sqrt','log2'],
                    'class_weight':['balanced',None]}
    grid_dt_acc = GridSearchCV(dt, param_grid = grid_values, cv = 5, verbose = 3.1, scoring = 'f1_macro')
    grid_dt_acc.fit(X_train, Y_train)
    print('Decision Tree Classifier:')
    print(grid_dt_acc.best_params_)
    x= grid_dt_acc.best_params_
    dt = DecisionTreeClassifier(criterion = x['criterion'],splitter = x['splitter'], max_features = x['max_features'],class_weight = x['class_weight'])
    dt.fit(X_train, Y_train)
    y_pred = dt.predict(X_test)
    scores(y_pred, Y_test)
    
# Description: Grid Search is performed on train set using Random Forest model to identify the best set of parameters that fits the data
def randomForestClassifier(X_train, X_test, Y_train, Y_test):
    rf = RandomForestClassifier(n_estimators = 300, max_features = 'log2', criterion = 'gini', bootstrap = True, max_samples = 0.5)
    from sklearn.model_selection import RandomizedSearchCV
    grid_values = {'n_estimators' : [100, 200,300, 400, 500],
                    'criterion' : ['gini', 'entropy'],
                    'bootstrap' : [True, False],
                    'max_features' : ['sqrt','log2'],
                    'max_samples' : [0.25, 0.5, 0.75] }
    grid_rf_acc = RandomizedSearchCV(rf, param_distributions = grid_values, cv = 5, verbose = 3.1, scoring = 'f1_macro')
    grid_rf_acc = GridSearchCV(rf, param_grid = grid_values, cv = 5, verbose = 3.1, scoring = 'f1_macro')
    grid_rf_acc.fit(X_train, Y_train)
    print('Random Forest Classifier')
    print(grid_rf_acc.best_params_)
    x = grid_rf_acc.best_params_
    rf = RandomForestClassifier(n_estimators = x['n_estimators'],max_features = x['max_features'],criterion = x['entropy'],bootstrap = x['bootstrap'], max_samples = x['max_samples'])
    rf.fit(X_train, Y_train)
    y_pred = rf.predict(X_test)
    scores(y_pred, Y_test)

# Description: Grid Search is performed on train set using KNN model to identify the best set of parameters that fits the data
def knn(X_train, X_test, Y_train, Y_test):
    clf = KNeighborsClassifier()
    grid_values = {'n_neighbors' : [5, 10, 15, 20, 25, 30, 45, 50],
                'weights' : ['distance','uniform'],
                'algorithm' : ['ball_tree','kd_tree','auto'],
                'metric' : ['euclidean','minkowski']}
    grid_knn_acc = GridSearchCV(clf, param_grid = grid_values, cv = 5, verbose = 3.1, scoring = 'f1_macro')
    grid_knn_acc.fit(X_train, Y_train)
    print('K Nearest Neighbors')
    print(grid_knn_acc.best_params_)
    x = grid_knn_acc.best_params_
    clf_knn = KNeighborsClassifier(n_neighbors = x['n_neighbors'],weights = x['weights'],algorithm = x['algorithm'],metric = x['metric'])
    clf_knn.fit(X_train, Y_train)
    y_pred = clf_knn.predict(X_test)
    scores(y_pred, Y_test)

# Description: Main function invokes all other functions
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

main()
