import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# generate the data
# generate random angles between [0, 2pi]
n = 800
rangle = 2 * np.pi * np.random.rand(n, 1)

# generate random radius for the first circle
e = 0.2
rr = 1.9 + e * np.random.rand(n, 1)
print("rr", rr)
rx = rr * np.sin(rangle)
ry = rr * np.cos(rangle)
print("rx", rx)
print("ry", ry)
x = rx
y = ry

# generate random radius for the second circle
rr2 = 1.2 + e * np.random.rand(n, 1)

rx2 = rr2 * np.sin(rangle)
ry2 = rr2 * np.cos(rangle)

x = np.concatenate((x, rx2))
y = np.concatenate((y, ry2))

rx3 = 1.4 + (1.9 - 1.4) * np.random.rand(10, 1)
ry3 = e * np.random.rand(10, 1)

# uncomment this to comment the two rings;
x = np.concatenate((x, rx3))
y = np.concatenate((y, ry3))

print("concatenated x", x)
print("concatenated y", y)

data = np.concatenate((x, y), axis=1)
print("data", data)

plt.scatter(data[:, 0], data[:, 1], c='black')
plt.title('original data')
plt.show()

# run kmeans on the original coordinates
K = 2
kmeans = KMeans(n_clusters=K).fit(data)
idx = kmeans.labels_
print("idx", idx)

data_r = data[np.where(idx == 0)]
data_b = data[np.where(idx == 1)]
print("len(data), len(data_r), len(data_b)", len(data), len(data_r), len(data_b))
print("data_r", data_r)
print("data_b", data_b)

plt.scatter(data_r[:, 0], data_r[:, 1], color='r')
plt.scatter(data_b[:, 0], data_b[:, 1], color='b')

plt.title('K-means plot')
plt.show()

distmat0 = pairwise_distances(data) 
distmat = pairwise_distances(data) * pairwise_distances(data)
print("distmat0.ndim, distmat0.size, distmat0.shape", distmat0.ndim, distmat0.size, distmat0.shape)
print("distmat.ndim, distmat.size, distmat.shape", distmat.ndim, distmat.size, distmat.shape)
print("distmat0", distmat0)
print("distmat", distmat)

A0 = (distmat < 0.1)                    #Boolean matrix
A = (distmat < 0.1).astype(np.int)      #Integer matrix
print("A0", A0)
print("A", A)

plt.spy(A)                      #can add markersize=0.1
plt.title('Adjacency Matrix')
plt.show()

D = np.diag(np.sum(A, axis=1))
print("D ", D)
L = D - A

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
