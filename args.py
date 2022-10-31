import argparse

def Parser1():
    parser = argparse.ArgumentParser(description='2_Kernel-based Learning')

    # data type
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--data-type', type=str, default='Diabetes.csv',
                        choices = ['abalone.csv', 'PersonalLoan.csv', 'WineQuality.csv', 'Diabetes.csv'])
    parser.add_argument('--reg-data', type=str, default='ToyotaCorolla.csv',
                        choices = ['Concrete.csv', 'Estate.csv', 'ToyotaCorolla.csv'])                   
    parser.add_argument('--seed', type=int, default=1592)              

    # Choose methods
    parser.add_argument('--method', type=str, default='SVR',
                        choices = ['Linearly_Hard', 'Linealry_Soft', 'Non_linearly_Soft', 'SVR'])

    # SVM & SVR Hyperparameters
    parser.add_argument('--split-size', type=float, default=0.2) # which means what percentage of learning data and validation data will be divided
    parser.add_argument('--C', type=float, default=5) # which means what percentage of learning data and validation data will be divided
    parser.add_argument('--kernel', type=str, default='poly',
                        choices = ['linear', 'rbf', 'poly', 'sigmoid'])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--degree', type=int, default=2)     

    return parser