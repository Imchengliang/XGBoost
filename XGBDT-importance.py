from XGBDT import model_training, data_processing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')

def drop(X_train, X_test, y_train, y_test, model, name):
        X_train1 = X_train.drop(name, axis=1)
        X_test1 = X_test.drop(name, axis=1)
        model.fit(X_train1, y_train.values.ravel())

        y_pred1 = model.predict(X_test1)

        accuracy = accuracy_score(y_test, y_pred1)
        print( 'accuracy:%2.f%%' %(accuracy*100), '\n')

        print(pd.crosstab(y_pred1, y_test), '\n')
        return y_pred1

def auc_roc(y_test, y_pred1, y_pred2, y_pred3):
        fpr_1, tpr_1, threshold_1 = roc_curve(y_test, y_pred1)  
        roc_auc_1 = auc(fpr_1, tpr_1) 
        fpr_2, tpr_2, threshold_2 = roc_curve(y_test, y_pred2)
        roc_auc_2 = auc(fpr_2, tpr_2)

        fpr_3, tpr_3, threshold_3 = roc_curve(y_test, y_pred3)
        roc_auc_3 = auc(fpr_3, tpr_3)

        plt.figure(figsize=(8, 5))
        plt.plot(fpr_1, tpr_1, color='darkorange', 
                lw=2, label='Without words of male and female (AUC = %0.3f)' % roc_auc_1, linestyle='-') 
        plt.plot(fpr_2, tpr_2, color='red',
                lw=2, label='Without released year (AUC = %0.3f)' % roc_auc_2, linestyle='--')
        plt.plot(fpr_3, tpr_3, color='green',
                lw=2, label='Without money made (AUC = %0.3f)' % roc_auc_3, linestyle='--')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.02, 1.05])
        plt.ylim([-0.02, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        #plt.savefig("hyh.png",dpi=600)
        plt.show()

if __name__ == "__main__":
        data = pd.read_csv("./train.csv")
        a, b, c, d, cor = data_processing(data, 0)
        e = model_training(a, b, c, d)
        f = drop(a, b, c, d, e, ['Number words male', 'Number words female'])
        g = drop(a, b, c, d, e, ['Year'])
        h = drop(a, b, c, d, e, ['Gross'])
        auc_roc(d, f, g, h)
