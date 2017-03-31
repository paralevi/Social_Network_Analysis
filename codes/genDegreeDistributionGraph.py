import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from numpy import *

def main():
    data = pd.read_csv('../YouTube-dataset/data/1-edges.csv',names=['u','v','w'])
    # generate undirected graph from data
    G = nx.Graph()
    G.add_edges_from(data[['u','v']].values.tolist())

    # get degree distribution
    degree = nx.degree_histogram(G)
    # x axis: degree k
    x = range(len(degree))
    # y axis: P(k)
    y = [z/float(G.number_of_nodes()) for z in degree]
    
    # remove 0 elements
    xx,yy=[],[]
    for (i,j) in zip(x,y):
        if i!=0 and j!=0:
            xx.append(i)
            yy.append(j)
    # linear fit using numpy.polyfit
    p = polyfit(log(xx),log(yy),1)
    y_hat=p[0]*log(xx)+p[1]

    # linear fit using Linear Regression, same result
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    tx=log(array(xx).reshape(-1,1))
    lr.fit(tx,log(yy))
    lr_y_predict = lr.predict(tx)

    #plotting figure
    plt.loglog(x,y,'.',xx,exp(lr_y_predict),'-')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.savefig('result.png')
    
    #R^2 metric
    from sklearn.metrics import r2_score
    print('R-squared Score:',r2_score(log(yy),y_hat))

    #K-S and P-value metric
    from scipy.stats import ks_2samp
    print(ks_2samp(log(yy),y_hat))
if __name__=='__main__':
    main()
