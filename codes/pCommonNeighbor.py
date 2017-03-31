import pandas as pd
import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.cross_validation import train_test_split
K = 10

def K_common_neighbor(G):
    l_size,m_size = G.number_of_edges(), max(G.nodes())
    E=G.edges()
    val = np.ones(l_size<<1,np.int8)
    col = np.zeros(l_size<<1,np.int16)  
    row = np.zeros(l_size<<1,np.int16)  
    # create adjacent sparse matrix
    for i in range(l_size):
        row[i<<1] = col[i<<1|1] = E[i][0]
        col[i<<1] = row[i<<1|1] = E[i][1]
    M = sparse.coo_matrix((val, (row, col)), shape=(m_size+1, m_size+1)).tocsr()
    '''
    calculate the number of common neighbors by matrix multiplication
    much faster than networkx built-in func
    '''
    M = M.dot(M)
    # mask = (1-np.identity(M.shape[0])) 
    print('dealing diagonal...')
    M.setdiag(0)
    M.eliminate_zeros()
    print('sorting...')
    pred = np.array(np.unravel_index(np.argsort((M.A).flatten(),axis=0)[-200:],M.shape)).T[::-1]
    res = []
    cnt = 0
    for u,v in pred:
        if (G.has_edge(u,v)):
            continue
        else:
            res.append((u,v))
            cnt = cnt+1
            if cnt == 2*K:
                break
    return res

def main():
    data = pd.read_csv('../YouTube-dataset/data/1-edges.csv',names=['u','v','w'])
    X_train,X_test,y_train,y_test=train_test_split(data[['u','v']],data['w'],test_size=0.1)
    G_train, G_test = nx.Graph(),nx.Graph()
    G_train.add_edges_from(X_train.values.tolist())
    G_test.add_edges_from(X_test.values.tolist())
    L = K_common_neighbor(G_train)
    correct,i = 0,0
    for u,v in L:
        i = i + 1
        if i % 2 == 0:
            continue
        print(u,v,end=' ')
        if G_test.has_edge(u,v):
            correct = correct+1
            print('T')
        else:
            print('F')
    print('Accuracy:',correct/(K))
if __name__=='__main__':
    main()
