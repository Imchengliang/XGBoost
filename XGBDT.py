import xgboost as xgb
from xgboost import plot_importance, plot_tree, plotting
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from dtreeviz import trees
from dtreeviz.models.xgb_decision_tree import ShadowXGBDTree
import graphviz
import pandas as pd
import numpy as np
import pandas_profiling
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')

def data_processing(data, z):
    # categorial features to number
    data["Lead"] = data["Lead"].map(lambda x: 1 if x == "Female" else 0)
    X = data.iloc[:, 0:13]
    
    corr = data.corr()
    leng = len(corr.columns)
    if z == 1:
        for elt in range(leng):
            i = 0
            while i < elt:
                if abs(corr.iloc[elt,i] > .7):
                    X=X.drop(columns=[corr.columns[elt]])
                i += 1
    
    col_name = X.columns
    # normalization
    X = preprocessing.normalize(X, norm='l2')
    X = pd.DataFrame(X, columns=col_name)
    y = data.iloc[:, 13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=70)
    return X_train, X_test, y_train, y_test, corr


    model5 = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        max_depth=5,
        min_child_weight=4,
        gamma=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=None,
        scale_pos_weight=1,
        seed=None)

    reg_alpha = [1e-5, 1e-2, 0.1, 1, 100]
    reg_lambda  =  [1e-5, 1e-2, 0.1, 1, 100]
    param5 = dict(reg_alpha=reg_alpha, reg_lambda=reg_lambda)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    grid_search = GridSearchCV(model5, param_grid=param5, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
    grid_result = grid_search.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

def model_training(X_train, X_test, y_train, y_test):
    #model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=200, verbosity=0, objective='binary:logistic')
    
    model = xgb.XGBClassifier(booster='gbtree', learning_rate=0.1, n_estimators=200,
        max_depth=5, min_child_weight=4, gamma=1, subsample=0.8, colsample_bytree=0.8,  
        reg_alpha=0.00001, reg_lambda=1,missing=None,nthread=None, seed=None, silent=None,
        random_state=0,  scale_pos_weight=1, verbosity=0, objective='binary:logistic')
    
    model.fit(X_train,y_train.values.ravel())

    # predict the label of test set
    y_pred = model.predict(X_test)

    # calculate the accuracy
    accuracy = accuracy_score(y_test,y_pred)
    print( 'accuracy:%2.f%%' %(accuracy*100), '\n')

    print(pd.crosstab(y_pred, y_test), '\n')

    plot_tree(model)

    # show feature importance
    xgb.plot_importance(model)
    plt.show()
    return model

# run cross-validation on train set and test set
def cv_score_train_test(X_train, X_test, y_train, y_test, model):
    num_cv = 10
    score_list = ["neg_log_loss","accuracy","f1", "roc_auc"]
    train_scores = []
    test_scores = []
    for score in score_list:
        train_scores.append(cross_val_score(model, X_train, y_train, cv=num_cv, scoring=score).mean())
        test_scores.append(cross_val_score(model, X_test, y_test, cv=num_cv, scoring=score).mean())
    scores = np.array((train_scores + test_scores)).reshape(2, -1)
    scores_df = pd.DataFrame(scores, index=['Train', 'Test'], columns=score_list)
    print(scores_df)


if __name__ == "__main__":
    data = pd.read_csv("./train.csv")
    a, b, c, d, cor = data_processing(data, 1)
    e = model_training(a, b, c, d)

