from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
import linkpred as lp
import cProfile
from time import clock
def main():
    data = pd.read_csv('../YouTube-dataset/data/1-edges.csv',names=['u','v','w'])
    X_train,X_test,y_train,y_test=train_test_split(data[['u','v']],data['w'],test_size=0.1)
    T = lp.link_prediction(X_train.values.tolist(),X_test.values.tolist())

    K=[10]
    for k in K:
        T.resource_allocation(k)
        print('K=%d,Precision=%.2f' % (k,T.test_precision(True)))

if __name__=='__main__':
    #cProfile.run('main()')
    start = clock()
    main()
    print(clock()-start)
