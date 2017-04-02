from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
import linkpred as lp
import cProfile
def main():
    data = pd.read_csv('../YouTube-dataset/data/1-edges.csv',names=['u','v','w'])
    X_train,X_test,y_train,y_test=train_test_split(data[['u','v']],data['w'],test_size=0.1)
    T = lp.link_prediction(X_train.values.tolist(),X_test.values.tolist())

    K=[10,20,30,50,100]
    for k in K:
        T.jaccard_coefficient(k)
        print('K=%d,Precision=%.2f' % (k,T.test_precision()))

if __name__=='__main__':
    #cProfile.run('main()')
    main()
