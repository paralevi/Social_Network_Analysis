import pandas as pd
import numpy as np
import heapq
from scipy import sparse
from sklearn.model_selection import train_test_split

class link_prediction(object):
    def __init__(self, E_train, E_test):
        node_size = 0

        for e in E_train:
            node_size = max(node_size,e[0],e[1])
        for e in E_test:
            node_size = max(node_size,e[0],e[1])

        self.M_train, self.adj_train = self._build_matrix(E_train,node_size)
        _,self.adj_test = self._build_matrix(E_test,node_size)

        self.pred = None
        self.predCN = None
        self.predJC = None
        self.predRA = None

    def _build_matrix(self, E, m_size):
        l_size = len(E)

        val = np.ones(l_size<<1,np.int8)
        col = np.zeros(l_size<<1,np.int16)
        row = np.zeros(l_size<<1,np.int16)
        # create adjacent sparse matrix
        for i,e in enumerate(E):
            u,v = e[0],e[1]
            row[i<<1] = col[i<<1|1] = u
            col[i<<1] = row[i<<1|1] = v

        M = sparse.coo_matrix((val, (row, col)), shape=(m_size+1, m_size+1))
        del val,col,row
        return M,M.toarray()

    def common_neighbors(self,K,range=200):
        if self.predCN is not None:
            self.pred = self.predCN[:K]
            return self.pred
        '''
        calculate the number of common neighbors by matrix multiplication
        much faster than networkx built-in func
        '''
        M2 = self.M_train.dot(self.M_train).tolil()
        M2.setdiag(0)
        row,col = M2.nonzero()
        index = zip(row,col)
        pred = list(filter(lambda e: not self.adj_train[e[1],e[2]] and e[1]<e[2],[(M2[u,v],u,v) for u,v in index]))
        self.predCN = list(map(lambda x:(x[1],x[2],x[0]), heapq.nlargest(range,pred)))
        del M2,pred,index,row,col
        self.pred = self.predCN[:K]
        return self.pred

    def jaccard_coefficient(self,K,range=200):
        if self.predJC is not None:
            self.pred = self.predJC[:K]
            return self.pred
        M2 = self.M_train.dot(self.M_train).tolil()
        M2.setdiag(0)
        t=self.M_train.sum(axis=0)
        row,col=M2.nonzero()
        index=zip(row,col)
        del row,col
        max_heap=[]
        for u,v in index:
            if M2[u,v]==0 or self.adj_train[u,v]:
                continue
            M2[v,u]=0
            jc = M2[u,v]/(t[0,i]+t[0,j]-M2[u,v])
            max_heap.append((jc,u,v))
        # pred = np.array(np.unravel_index(np.argsort((M.A).flatten(),axis=0)[-K:],M.shape)).T[::-1]
        self.predJC = list(map(lambda x:(x[1],x[2],x[0]), heapq.nlargest(range,max_heap)))
        self.pred = self.predJC[:K]
        return self.pred

    def resource_allocation(self,K,range=200):
        if self.predRA is not None:
            self.pred = self.predRA[:K]
            return self.pred
        M2 = self.M_train.dot(self.M_train).tolil()
        M2.setdiag(0)
        t=self.M_train.sum(axis=0)
        t=1/np.array(t)
        row,col=M2.nonzero()
        index=zip(row,col)
        del row,col
        max_heap=[]
        x=0
        for u,v in index:
            if M2[u,v]==0 or self.adj_train[u,v]:
                continue
            M2[v,u]=0
            if u != x:
                x=u
                print(u)
            ra = sum((self.adj_train[u]&self.adj_train[v])*t[0])
            max_heap.append((ra,u,v))
        self.predRA = max_heap
        print(max_heap)
        self.predRA = list(map(lambda x:(x[1],x[2],x[0]), heapq.nlargest(range if len(max_heap)<range else len(max_heap),max_heap)))
        self.pred = self.predRA[:K]
        return self.pred

    def test_precision(self,detail=False):
        L = self.pred
        correct = 0
        for u,v,c in L:
            check = self.adj_test[u,v] != 0
            if detail == True:
                print(u,v,c,check)
            if check == True:
                correct = correct+1
        precision = correct/len(L)
        return precision
