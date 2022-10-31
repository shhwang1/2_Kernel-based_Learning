import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Linearly_Hard_Case(args):
    data = pd.read_csv(args.data_path + args.data_type)
    h = .02
    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = args.split_size, shuffle=True, random_state = args.seed)

    clf = svm.SVC(kernel='linear') # 1e-10 means very small value which is close to "0" (There's no Hard-margin method in sklearn)
    clf.fit(X_train, y_train)
    print('Training SVM : Linearly-Hard Margin SVM.....')

    # Prediction
    pred_clf = clf.predict(X_test)
    accuracy_clf = accuracy_score(y_test, pred_clf)
    print('Linearly-Hard margin SVM accuracy : ', accuracy_clf)





