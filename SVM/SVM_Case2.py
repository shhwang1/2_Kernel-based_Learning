import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Linearly_Soft_Case(args):
    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = args.split_size, shuffle=True, random_state = args.seed)

    clf = svm.SVC(C = args.C, kernel='linear') # C is bigger than 0 because error is allowed.
    clf.fit(X_train, y_train)
    print('Training SVM : Linearly-Soft Margin SVM.....')

    # Prediction
    pred_clf = clf.predict(X_test)
    accuracy_clf = accuracy_score(y_test, pred_clf)
    print('Linearly-Soft margin SVM accuracy : ', accuracy_clf)
