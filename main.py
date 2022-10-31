from args import Parser1
from SVM.SVM_Case1 import Linearly_Hard_Case
from SVM.SVM_Case2 import Linearly_Soft_Case
from SVM.SVM_Case4 import Non_linearly_Soft_Case
from SVR.SupportVectorRegression import SVR

def build_model():
    parser = Parser1()
    args = parser.parse_args()

    if args.method == 'Lineary_Hard':
        model = Linearly_Hard_Case(args)
    elif args.method == 'Linearly_Soft':
        model = Linearly_Soft_Case(args)
    elif args.method == 'Non_linearly_Soft':
        model = Non_linearly_Soft_Case(args)
    else:
        model = SVR(args)

    return model

def main():
    build_model()

if __name__ == '__main__':
    main()
