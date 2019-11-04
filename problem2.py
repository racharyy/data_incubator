import numpy as np
from itertools import product
from copy import copy
from scipy.sparse import csr_matrix
import pickle

def lin_ind(ind,n):
    x = 0 
    d= len(ind)
    for i,j in enumerate(ind):
        assert(j<n) 
        x=x+j*n**(d-1-i)
    return x


def adj_matrix(d,n):
    l = [range(n) for dim in range(d)]
    mat = np.zeros((n**d,n**d))
    for ind in product(*l):
        cur_flat = lin_ind(ind,n)
        for i in range(d):
            cur= list(copy(ind))
            if ind[i]<n-1:
                cur[i] = cur[i]+1
                #print(cur)
                flat_ind = lin_ind(cur,n)
                
                mat[cur_flat][flat_ind]=1
                mat[flat_ind][cur_flat]=1
    return mat
            
def sparse_adj_matrix(d,n):
    l = [range(n) for dim in range(d)]
    tot_len = 2*d*(n**(d-1))*(n-1)
    row_ind, col_ind, data = np.zeros(tot_len),np.zeros(tot_len),np.ones(tot_len)
    iter_num =0

    for ind in product(*l):
        if iter_num%100000==0:
            print(iter_num)
        cur_flat = lin_ind(ind,n)
        for i in range(d):
            cur= list(copy(ind))
            if ind[i]<n-1:
                cur[i] = cur[i]+1
                #print(cur)
                flat_ind = lin_ind(cur,n)
                row_ind[2*iter_num] = cur_flat
                row_ind[2*iter_num+1] = flat_ind
                col_ind[2*iter_num] = flat_ind
                col_ind[2*iter_num+1] = cur_flat
                
                iter_num = iter_num +1
    return row_ind, col_ind, data
d=8
n=12
m=12
row_ind, col_ind, data = sparse_adj_matrix(d,n)
with open('row_col_data.pkl','wb') as f:
    pickle.dump((row_ind, col_ind, data),f)
row_ind, col_ind, data = pickle.load(open('row_col_data.pkl','rb'))
mat = csr_matrix((data, (row_ind, col_ind)), shape=(n**d, n**d))
with open('n12_d8_mat.pkl','wb') as f1:
    pickle.dump(mat,f1)
#mat = mat.dot(mat)
mat_s = mat
for i in range(m-1):
    print(i)
    mat_s = mat_s.dot(mat)
with open('n12_d8_mat_product.pkl','wb') as f2:
    pickle.dump(mat_s,f2)
print(mat_s.sum(axis=0))




