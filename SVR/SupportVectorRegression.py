import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def SVR(args):
    data = pd.read_csv(args.data_path + args.reg_data)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = args.split_size, shuffle=True, random_state = args.seed)

    if args.kernel == 'rbf':
        clf = svm.SVR(C = args.C, kernel = args.kernel, gamma = args.gamma) # C is bigger than 0 because error is allowed.
        clf.fit(X_train, y_train)
        print('Training SVR with ' + args.kernel + '.....')

    elif args.kernel == 'poly':
        clf = svm.SVR(C = args.C, kernel = args.kernel, degree = args.degree) # C is bigger than 0 because error is allowed.
        clf.fit(X_train, y_train)
        print('Training SVR with ' + args.kernel + '.....')

    else:
        clf = svm.SVR(C = args.C, kernel = args.kernel) # C is bigger than 0 because error is allowed.
        clf.fit(X_train, y_train)
        print('Training SVR with ' + args.kernel + '.....')

    # Prediction
    pred_clf = clf.predict(X_test)
    reg_r2_score = r2_score(y_test, pred_clf)
    reg_mae_score = mean_squared_error(y_test, pred_clf)
    reg_mape_score = mean_absolute_percentage_error(y_test, pred_clf)

    print('SVR with ' + args.kernel + 'R2 Score : ', reg_r2_score)
    print('SVR with ' + args.kernel + 'MAE Score : ', reg_mae_score)
    print('SVR with ' + args.kernel + 'MAPE Score : ', reg_mape_score)
