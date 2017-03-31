import pandas as pd
import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.cross_validation import train_test_split
import heapq
K = 20

def jaccard_coefficient(G):
    l_size,m_size = G.number_of_edges(), max(G.nodes())
    E=G.edges()
    val = np.ones(l_size<<1,np.float16)
    col = np.zeros(l_size<<1,np.int16)  
    row = np.zeros(l_size<<1,np.int16)  
    for i in range(l_size):
        row[i<<1] = col[i<<1|1] = E[i][0]
        col[i<<1] = row[i<<1|1] = E[i][1]
    M = sparse.coo_matrix((val, (row, col)), shape=(m_size+1, m_size+1))
    M2 = M.dot(M).tolil()
    M2.setdiag(0)
    t=M.sum(axis=0)
    row,col=M2.nonzero()
    index=zip(row,col)
    del row,col
    max_heap=[]
    M = sparse.lil_matrix(M.shape)
    print('calculating...')
    for i,j in index:
        if M2[i,j] == 0 or G.has_edge(i,j):
            continue
        M2[j,i] = 0
        M[i,j]=M2[i,j]/(t[0,i]+t[0,j]-M2[i,j])
        if len(max_heap) <= K:
            heapq.heappush(max_heap,(-M[i,j],i,j))
        else:
            heapq.heappushpop(max_heap,(-M[i,j],i,j))
    # pred = np.array(np.unravel_index(np.argsort((M.A).flatten(),axis=0)[-K:],M.shape)).T[::-1]
    return list(map(lambda x:(x[1],x[2],-x[0]), max_heap))

def networkx(G):
    max_heap=[]
    L=nx.jaccard_coefficient(G)
    for u,v,j in L:
        if G.has_edge(u,v):
            continue
        if len(max_heap) <= K:
            heapq.heappush(max_heap,(-j,u,v))
        else:
            heapq.heappushpop(max_heap,(-j,u,v))
    return list(map(lambda x:(x[1],x[2],-x[0]), max_heap))

def main():
    data = pd.read_csv('../YouTube-dataset/1-edges.csv',names=['u','v','w'])
    X_train,X_test,y_train,y_test=train_test_split(data[['u','v']],data['w'],test_size=0.1)

    G_train, G_test = nx.Graph(),nx.Graph()
    G_train.add_edges_from(X_train.values.tolist())
    G_test.add_edges_from(X_test.values.tolist())

    L = jaccard_coefficient(G_train)
    '''
    very slow! more than half an hour in test
    L = networkx(G_train)
    '''
    correct,i = 0,0
    for u,v,j in L:
        print(u,v,j,end=' ')
        if G_test.has_edge(u,v):
            correct = correct+1
            print('T')
        else:
            print('F')
    print('Accuracy:',correct/len(L))

if __name__=='__main__':
    main()
