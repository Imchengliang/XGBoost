from XGBDT import model_training, data_processing
from sklearn import preprocessing
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')

def real_test(df, model, cor):
    # preprocess test data
    df1 = df
    corr = cor
    leng = len(corr.columns)
    for elt in range(leng):
        i = 0
        while i < elt:
            if abs(corr.iloc[elt,i] > .7):
                df = df.drop(columns=[corr.columns[elt]])
            i += 1
    col_name = df.columns
    df = preprocessing.normalize(df, norm='l2')
    df = pd.DataFrame(df, columns=col_name)
    # predict the test data, write the prediction into CSV, draw a diagram about the prediction
    prediction = model.predict(df)
    with open("./predict_result.csv", 'w') as f:
        wr = csv.writer(f)
        wr.writerow(prediction)
    l = [0, 0]
    for i in prediction:
        if i == 0:
            l[0] = l[0] + 1
        elif i == 1:
            l[1] = l[1] + 1
    candidate = ['male', 'female']
    explode = (0.1, 0)
    plt.figure(figsize=(10, 10), dpi=100)
    plt.pie(l, labels=candidate, autopct="%1.2f%%", colors=['c', 'm'],
            textprops={'fontsize': 24}, labeldistance=1.05, explode=explode, startangle=90, shadow=True)
    plt.legend(fontsize=16)
    plt.title("test result", fontsize=24)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("./test.csv")
    data = pd.read_csv("./train.csv")
    a, b, c, d, cor = data_processing(data, 1)
    e = model_training(a, b, c, d)
    real_test(df, e, cor)
