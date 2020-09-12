import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


file_path = os.path.abspath('nodes.txt')
nodes_names = {}
nodes_labels = dict()
if os.path.exists(file_path):
    with open(file_path, 'r') as f_h:   #https://cmdlinetips.com/2016/01/opening-a-file-in-python-using-with-statement/
        for line in f_h.readlines():
            words = line.split("\t", 4)
            nodes_names[int(words[0])-1] = words[1]
            nodes_labels[int(words[0])-1] = int(words[2])       #nodes go from 0 to 1489; so offset by -1 from original numbering

print(len(nodes_names), nodes_names)
print(len(nodes_labels), nodes_labels)

A = np.zeros([len(nodes_labels), len(nodes_labels)], dtype=int)

file_path = os.path.abspath('edges.txt')
if os.path.exists(file_path):
    with open(file_path, 'r') as f_h:
        for line in f_h.readlines():
            words = line.rstrip().split("\t", 2)
            x1, x2 = int(words[0]), int(words[1])
            A[x1-1, x2-1]=1
            A[x2-1, x1-1] = np.copy(A[x1-1, x2-1])

print("A.ndim, A.shape, A.size", A.ndim, A.shape, A.size)
print("A", A)

D0 = np.diag(np.sum(A, axis=1))
print("D0 ", D0)
print("D0.ndim, D0.shape, D0.size", D0.ndim, D0.shape, D0.size)
for i in range(D0.shape[0]):
    print(D0[i,i])

D = np.diag(1/np.sqrt(np.sum(A, axis=1)))      #https://numpy.org/doc/stable/reference/generated/numpy.sum.html
                                    #https://numpy.org/doc/stable/reference/generated/numpy.diag.html
print("D ", D)
print("D.ndim, D.shape, D.size", D.ndim, D.shape, D.size)

'''
L = D - A
print("L.ndim, L.shape, L.size", L.ndim, L.shape, L.size)
print("L ", L)
print("L[0, :] ", L[0, :])
print("L[1, :] ", L[1, :])
print("L[1488, :] ", L[1488, :])
print("L[1489, :] ", L[1489, :])
print("sum(L[0, :]) ", sum(L[0, :]))
print("sum(L[1, :]) ", sum(L[1, :]))
print("sum(L[1488, :]) ", sum(L[1488, :]))
print("sum(L[1489, :]) ", sum(L[1489, :]))

eig_w, eig_v = np.linalg.eig(L)
print("ndim, shape, eig_w of L", eig_w.ndim, eig_w.shape, "\n", eig_w)
print("ndim, shap, eig_v of L", eig_v.ndim, eig_v.shape, "\n", eig_v)

#Extract the real components
eig_w = eig_w.real
eig_v = eig_v.real

eig_w_2d = eig_w.reshape(1, eig_w.shape[0])
print("eig_w_2d \n", eig_w_2d)
print("eig_v \n", eig_v)
print("eig_v[0,0], eig_v[0,1], eig_v[1,0]", eig_v[0,0], eig_v[0,1], eig_v[1,0])
print("eig_v[:,0], eig_v[:,1] \n", eig_v[:,0], "\n", eig_v[:,1])
print("eig_v[:, 0:2]", eig_v[:, 0:2])
print("eig_v[:, [0,1]]", eig_v[:, [0,1]])

eig_w_v = np.concatenate((eig_w_2d, eig_v), axis=0)   #row0 are eig_w. Each col is:  one eig_w + corresponding eig_v
print("eig_w_v[:,0], eig_w_v[:,1], eig_w_v[:,2] \n", eig_w_v[:,0], "\n", eig_w_v[:,1], "\n", eig_w_v[:,2])

print("eig_w_v[:,1489], eig_w_v[:,1488], eig_w_v[:,1487] \n", eig_w_v[:,1489], "\n", eig_w_v[:,1488], "\n", eig_w_v[:,1487])


#Sort all cols based on values in row0/eig_values   https://numpy.org/doc/stable/reference/generated/numpy.argsort.html 
#https://www.kite.com/python/answers/how-to-sort-the-rows-of-a-numpy-array-by-a-column-in-python

eig_w_v_sorted =  eig_w_v[:, np.argsort(eig_w_v[0,:])]
print("Sorted eigen values eig_w_v_sorted[0,:] \n", eig_w_v_sorted[0, :])
print("eig_w_v_sorted[:,0], eig_w_v_sorted[:,1], eig_w_v_sorted[:,2] eig_w_v_sorted[:,3] \n", \
    eig_w_v_sorted[:,0], "\n", eig_w_v_sorted[:,1], "\n", eig_w_v_sorted[:,2], "\n", eig_w_v_sorted[:,3])

print("eig_w_v_sorted[:,200], eig_w_v_sorted[:,201], eig_w_v_sorted[:,202] eig_w_v_sorted[:,203] \n", \
    eig_w_v_sorted[:,200], "\n", eig_w_v_sorted[:,201], "\n", eig_w_v_sorted[:,202], "\n", eig_w_v_sorted[:,203])

print("eig_w_v_sorted[:,250], eig_w_v_sorted[:,251], eig_w_v_sorted[:,252] eig_w_v_sorted[:,253] \n", \
    eig_w_v_sorted[:,250], "\n", eig_w_v_sorted[:,251], "\n", eig_w_v_sorted[:,252], "\n", eig_w_v_sorted[:,253])

print("eig_w_v_sorted[:,300], eig_w_v_sorted[:,301], eig_w_v_sorted[:,302] eig_w_v_sorted[:,303] \n", \
    eig_w_v_sorted[:,300], "\n", eig_w_v_sorted[:,301], "\n", eig_w_v_sorted[:,302], "\n", eig_w_v_sorted[:,303])

print("eig_w_v_sorted[:,350], eig_w_v_sorted[:,351], eig_w_v_sorted[:,352] eig_w_v_sorted[:,353] \n", \
    eig_w_v_sorted[:,350], "\n", eig_w_v_sorted[:,351], "\n", eig_w_v_sorted[:,352], "\n", eig_w_v_sorted[:,353])

print("eig_w_v_sorted[:,400], eig_w_v_sorted[:,401], eig_w_v_sorted[:,402] eig_w_v_sorted[:,403] \n", \
    eig_w_v_sorted[:,400], "\n", eig_w_v_sorted[:,401], "\n", eig_w_v_sorted[:,402], "\n", eig_w_v_sorted[:,403])

print("eig_w_v_sorted[:,450], eig_w_v_sorted[:,451], eig_w_v_sorted[:,452] eig_w_v_sorted[:,453] \n", \
    eig_w_v_sorted[:,450], "\n", eig_w_v_sorted[:,451], "\n", eig_w_v_sorted[:,452], "\n", eig_w_v_sorted[:,453])

print("eig_w_v_sorted[:,1489], eig_w_v_sorted[:,1488], eig_w_v_sorted[:,1487] eig_w_v_sorted[:,1486] \n", \
    eig_w_v_sorted[:,1489], "\n", eig_w_v_sorted[:,1488], "\n", eig_w_v_sorted[:,1487], "\n", eig_w_v_sorted[:,1486])

K = input("Enter the number of clusters to be formed : ")
try:
    K = int(K)
except:
    print("bad K = ", K, "and so K is being assigned K=2")
    K=2
print("K = ", K)

#Now choose the first K 


###########

s, v = np.linalg.eig(L)
print("ndim, shape, eigenvalues of L", s.ndim, s.shape, s)
print("ndim, shap, eigenvectors of L", v.ndim, v.shape, v)
K = 2
v = v[:, 0:K].real
kmeans = KMeans(n_clusters=K).fit(v)
idx = kmeans.labels_

data_g = data[np.where(idx==0)]
data_m = data[np.where(idx==1)]

plt.scatter(data_g[:, 0], 
            data_g[:, 1],
            c='g')

plt.scatter(data_m[:, 0], 
            data_m[:, 1],
            c='m')

plt.title('Spectral Clustering')
plt.show()
'''