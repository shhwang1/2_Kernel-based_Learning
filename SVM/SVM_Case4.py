import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Non_linearly_Soft_Case(args):
    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = args.split_size, shuffle=True, random_state = args.seed)

    if args.kernel == 'rbf':
        clf = svm.SVC(C = args.C, kernel = args.kernel, gamma = args.gamma) # C is bigger than 0 because error is allowed.
        clf.fit(X_train, y_train)
        print('Training SVM : Non_linearly-Soft Margin SVM with ' + args.kernel + '.....')

    elif args.kernel == 'poly':
        clf = svm.SVC(C = args.C, kernel = args.kernel, degree = args.degree) # C is bigger than 0 because error is allowed.
        clf.fit(X_train, y_train)
        print('Training SVM : Non_linearly-Soft Margin SVM with ' + args.kernel + '.....')

    else:
        clf = svm.SVC(C = args.C, kernel = args.kernel) # C is bigger than 0 because error is allowed.
        clf.fit(X_train, y_train)
        print('Training SVM : Non_linearly-Soft Margin SVM with ' + args.kernel + '.....')

    # Prediction
    pred_clf = clf.predict(X_test)
    accuracy_clf = accuracy_score(y_test, pred_clf)
    print('Non_linearly-Soft margin SVM with ' + args.kernel + 'accuracy : ', accuracy_clf)
