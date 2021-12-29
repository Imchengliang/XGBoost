from XGBDT import model_training, data_processing
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def learning_rate_n_estimators(X_train, X_test, y_train, y_test):
    model1 = xgb.XGBClassifier(
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=None,
        scale_pos_weight=1,
        seed=None)

    learning_rate = [ 0.001, 0.01, 0.1, 0.2]
    n_estimators = [100, 200, 300, 500, 1000]
    param1 = dict(learning_rate=learning_rate, n_estimators=n_estimators)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    # GridSearchCV requires param_grid to be a dict or a list and scoring can be roc_auc or neg_log_loss
    grid_search = GridSearchCV(model1, param_grid=param1, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
    grid_result = grid_search.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

def max_depth_min_child_weight(X_train, X_test, y_train, y_test):
    model2 = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=None,
        scale_pos_weight=1,
        seed=None)

    max_depth = [ i for i in range(1, 6)]
    min_child_weight = [i for i in range(4, 8)]
    param2 = dict(max_depth=max_depth, min_child_weight=min_child_weight)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    grid_search = GridSearchCV(model2, param_grid=param2, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
    grid_result = grid_search.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

def gamma(X_train, X_test, y_train, y_test):
    model3 = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        max_depth=5,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=None,
        scale_pos_weight=1,
        seed=None)

    gamma = [ i/10.0 for i in range(5, 12)]
    param3 = dict(gamma=gamma)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    grid_search = GridSearchCV(model3, param_grid=param3, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
    grid_result = grid_search.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

def subsample_colsample_bytree(X_train, X_test, y_train, y_test):
    model4 = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        max_depth=5,
        min_child_weight=4,
        gamma=1,
        objective= 'binary:logistic',
        nthread=None,
        scale_pos_weight=1,
        seed=None)

    subsample = [ i/10.0 for i in range(6, 10)]
    colsample_bytree  =  [ i/10.0 for i in range(6, 10)]
    param4 = dict(subsample=subsample, colsample_bytree=colsample_bytree)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    grid_search = GridSearchCV(model4, param_grid=param4, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
    grid_result = grid_search.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

def reg_alpha_reg_lambda(X_train, X_test, y_train, y_test):
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

if __name__ == "__main__":
    data = pd.read_csv("./train.csv")
    a, b, c, d, cor = data_processing(data)
    learning_rate_n_estimators(a, b, c, d)
    max_depth_min_child_weight(a, b, c, d)
    gamma(a, b, c, d)
    subsample_colsample_bytree(a, b, c, d)
    reg_alpha_reg_lambda(a, b, c, d)