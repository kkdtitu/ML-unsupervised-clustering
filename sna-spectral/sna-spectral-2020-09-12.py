import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


file_path = os.path.abspath('nodes.txt')
nodes_nums = list()
nodes_names = []
nodes_labels = []
if os.path.exists(file_path):
    with open(file_path, 'r') as f_h:   #https://cmdlinetips.com/2016/01/opening-a-file-in-python-using-with-statement/
        for line in f_h.readlines():
            words = line.split("\t", 4)
            nodes_nums.append(int(words[0]))
            nodes_names.append(words[1])
            nodes_labels.append(int(words[2]))       #nodes go from 0 to 1489; so offset by -1 from original numbering

print("nodes_nums",nodes_nums)
print(len(nodes_names), "nodes_names",nodes_names)
print(len(nodes_labels), "nodes_labels",nodes_labels)

nodes_if_connected = list()
for i in range(len(nodes_nums)):
    nodes_if_connected.append(0)
file_path = os.path.abspath('edges.txt')
if os.path.exists(file_path):
    with open(file_path, 'r') as f_h:
        for line in f_h.readlines():
            words = line.rstrip().split("\t", 2)
            x1, x2 = int(words[0]), int(words[1])
            nodes_if_connected[nodes_nums.index(x1)]=1
            nodes_if_connected[nodes_nums.index(x2)]=1

print("nodes_if_connected: ", nodes_if_connected)

nodes_nums_adj = list()
for i in range(len(nodes_nums)):
    if nodes_if_connected[i]:
        nodes_nums_adj.append(nodes_nums[i])
print("len(nodes_nums_adj), nodes_nums_adj", len(nodes_nums_adj), nodes_nums_adj)

A = np.zeros([len(nodes_nums_adj), len(nodes_nums_adj)], dtype=int)

file_path = os.path.abspath('edges.txt')
if os.path.exists(file_path):
    with open(file_path, 'r') as f_h:
        for line in f_h.readlines():
            words = line.rstrip().split("\t", 2)
            x1, x2 = int(words[0]), int(words[1])
            x1_index = nodes_nums_adj.index(x1)
            x2_index = nodes_nums_adj.index(x2)
            A[x1_index, x2_index]=1
            A[x2_index, x1_index] = np.copy(A[x1_index, x2_index])

print("A.ndim, A.shape, A.size", A.ndim, A.shape, A.size)
print("A", A)

D = np.diag(np.sum(A, axis=1))             #https://numpy.org/doc/stable/reference/generated/numpy.sum.html
                                            #https://numpy.org/doc/stable/reference/generated/numpy.diag.html
print("D ", D)
print("D.ndim, D.shape, D.size", D.ndim, D.shape, D.size)
sum_0d = 0 
for i in range(D.shape[0]):
    print("D i ", i, D[i,i])
    if not D[i,i]:
        sum_0d+=1
print("sum_0d ", sum_0d)
#D = np.diag(1/np.sqrt(np.sum(A, axis=1)))      #https://numpy.org/doc/stable/reference/generated/numpy.sum.html
#                                              #https://numpy.org/doc/stable/reference/generated/numpy.diag.html
#print("D ", D)
#print("D.ndim, D.shape, D.size", D.ndim, D.shape, D.size)


L = D - A
print("L.ndim, L.shape, L.size", L.ndim, L.shape, L.size)
print("L ", L)
print("L[0, :] ", L[0, :])
print("L[1, :] ", L[1, :])
print("L[1188, :] ", L[1188, :])
print("L[1189, :] ", L[1189, :])
print("sum(L[0, :]) ", sum(L[0, :]))
print("sum(L[1, :]) ", sum(L[1, :]))
print("sum(L[1188, :]) ", sum(L[1188, :]))
print("sum(L[1189, :]) ", sum(L[1189, :]))

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

print("eig_w_v[:,1189], eig_w_v[:,1188], eig_w_v[:,1187] \n", eig_w_v[:,1189], "\n", eig_w_v[:,1188], "\n", eig_w_v[:,1187])


#Sort all cols based on values in row0/eig_values   https://numpy.org/doc/stable/reference/generated/numpy.argsort.html 
#https://www.kite.com/python/answers/how-to-sort-the-rows-of-a-numpy-array-by-a-column-in-python

eig_w_v_sorted =  eig_w_v[:, np.argsort(eig_w_v[0,:])]
print("Sorted eigen values eig_w_v_sorted[0,:] \n", eig_w_v_sorted[0, :])
print("eig_w_v_sorted[:,0], eig_w_v_sorted[:,1], eig_w_v_sorted[:,2] eig_w_v_sorted[:,3] \n", \
    eig_w_v_sorted[:,0], "\n", eig_w_v_sorted[:,1], "\n", eig_w_v_sorted[:,2], "\n", eig_w_v_sorted[:,3])

print("eig_w_v_sorted[:,4], eig_w_v_sorted[:,5], eig_w_v_sorted[:6] eig_w_v_sorted[:,7] \n", \
    eig_w_v_sorted[:,4], "\n", eig_w_v_sorted[:,5], "\n", eig_w_v_sorted[:,6], "\n", eig_w_v_sorted[:,7])

print("eig_w_v_sorted[:,250], eig_w_v_sorted[:,251], eig_w_v_sorted[:,252] eig_w_v_sorted[:,253] \n", \
    eig_w_v_sorted[:,250], "\n", eig_w_v_sorted[:,251], "\n", eig_w_v_sorted[:,252], "\n", eig_w_v_sorted[:,253])


print("eig_w_v_sorted[:,1220], eig_w_v_sorted[:,1221], eig_w_v_sorted[:,1222] eig_w_v_sorted[:,1223] \n", \
    eig_w_v_sorted[:,1220], "\n", eig_w_v_sorted[:,1221], "\n", eig_w_v_sorted[:,1222], "\n", eig_w_v_sorted[:,1223])



K = input("Enter the number of clusters to be formed : ")
try:
    K = int(K)
except:
    print("bad K = ", K, "and so K is being assigned K=2")
    K=2
print("K = ", K)

#Now choose the first K 


###########
eig_w_v_sorted_K = eig_w_v_sorted[1:, 0:K]
print("eig_w_v_sorted_K.shape, eig_w_v_sorted[:0], eig_w_v_sorted[:1], eig_w_v_sorted[:2],", \
    eig_w_v_sorted_K.shape, eig_w_v_sorted[:,0], eig_w_v_sorted[:,1], eig_w_v_sorted[:,2])

kmeans = KMeans(n_clusters=K).fit(eig_w_v_sorted_K)
idx = kmeans.labels_

count_cluster_not_0 = 0
for i in range(len(idx)):
    print("idx i", i, idx[i])
    if idx[i]:
        count_cluster_not_0 +=1

print("count_cluster_not_0 ", count_cluster_not_0 )
'''
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