import skimage
import sklearn
import sklearn_extra
from skimage import io
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np

#Read the image into an np array (3D by default) 
image_orig = skimage.io.imread('beach.bmp')
io.imshow(image_orig)
io.show()

#Dimension of the original image / np array
print("type(image_orig)", type(image_orig))
print("image_orig.shape", image_orig.shape)
print("image_orig: Number of dimensions, rows,cols, deep, total-size, length-of-1st-dim ", \
    image_orig.ndim, image_orig.shape[0], image_orig.shape[1], image_orig.shape[2], image_orig.size, len(image_orig))
print("Top left image_orig[0][0]", image_orig[0][0])
print("Top left image_orig[0, 0]", image_orig[0, 0])
print("Bottom right image_orig[213][319]", image_orig[213][319])
print("Bottom right image_orig[213, 319]", image_orig[213, 319])

#Flatten the original image; dimension of the flattened image / np array
rows = image_orig.shape[0]
cols = image_orig.shape[1]
image_f = image_orig.reshape(rows*cols, 3)
print("image_f.shape", image_f.shape)
print("image_f : Number of dimensions, rows, deep, total-size, length-of-1st-dim ", image_f.ndim, image_f.shape[0], image_f.shape[1], \
    image_f.size, len(image_f))
print("image_f[0]", image_f[0])
print("image_f[68479]", image_f[68479])  #214*320-1


#create a sample np array that samples every 5th pixel
image_f_s = np.empty([13590, 3])   #68480/5 ~ 13590
#print("Empty image_f_s: ", image_f_s)
for i in range(13590):
    image_f_s[i] = image_f[i*5]
print("Sampled image_f_s: ", image_f_s)
print("image_f_s: Number of dimensions, rows, deep, total-size, length-of-1st-dim ", image_f_s.ndim, image_f_s.shape[0], image_f_s.shape[1], \
    image_f_s.size, len(image_f_s))

#Implement k-means clustering to form k clusters from image_f / original flattened image / 2D np array
kmeans = sklearn.cluster.KMeans(n_clusters=16)
kmeans_image_f = kmeans.fit(image_f)
print("kmeans_image_f", kmeans_image_f)
#print("kmeans_image_f.shape [0] [1]", kmeans_image_f.shape[0], kmeans_image_f.shape[1])

#Replace each pixel value with its nearby centroid and creating image_f_c_with_kmeans / 2D np array 
print("len(kmeans_image_f.cluster_centers_) ", len(kmeans_image_f.cluster_centers_))
print("kmeans_image_f.cluster_centers_ ", kmeans_image_f.cluster_centers_)
print("len(kmeans_image_f.labels_)" , len(kmeans_image_f.labels_))
print("kmeans_image_f.labels_" , kmeans_image_f.labels_)
image_f_c_with_kmeans = kmeans_image_f.cluster_centers_[kmeans_image_f.labels_]
print("image_f_c_with_kmeans shape: ", image_f_c_with_kmeans.shape)
rows_c1 = image_f_c_with_kmeans.shape[0]
cols_c1 = image_f_c_with_kmeans.shape[1]
print("image_f_c_with_kmeans rows, cols: ", rows_c1,cols_c1)
print("image_f_c_with_kmeans[0]", image_f_c_with_kmeans[0])

image_f_c_with_kmeans_clip = np.clip(image_f_c_with_kmeans.astype('uint8'), 0, 255)
print("image_f_c_with_kmeans_clip[0]", image_f_c_with_kmeans_clip[0])

#Reshape image_f_c_with_kmeans_clip to original dimension
image_f_c_with_kmeans_clip_final = image_f_c_with_kmeans_clip.reshape(rows, cols, 3)
print("image_f_c_with_kmeans_clip_final.shape", image_f_c_with_kmeans_clip_final.shape)
print("image_f_c_with_kmeans_clip_final.shape [0][1][2]", \
    image_f_c_with_kmeans_clip_final.shape[0], image_f_c_with_kmeans_clip_final.shape[1], image_f_c_with_kmeans_clip_final.shape[2])
print(image_f_c_with_kmeans_clip_final)

#Save and display output image
skimage.io.imsave('beach_c_with_kmeans.png', image_f_c_with_kmeans_clip_final)
io.imshow(image_f_c_with_kmeans_clip_final)
io.show()



#Implement k-medoids clustering to form k clusters from image_f_s / flattened and sampled image / 2D np array
kmedoids = sklearn_extra.cluster.KMedoids(n_clusters=16)
kmedoids_image_f_s = kmedoids.fit(image_f_s)
print("kmedoids_image_f_s", kmedoids_image_f_s)

#Replace each pixel value with its nearby centroid and creating image_f_s_c_with_kmedoids / 2D np array 
image_f_s_c_with_kmedoids = kmedoids_image_f_s.cluster_centers_[kmedoids_image_f_s.labels_]
print("image_f_s_c_with_kmedoids shape: ", image_f_s_c_with_kmedoids.shape)
rows_c2 = image_f_s_c_with_kmedoids.shape[0]
cols_c2 = image_f_s_c_with_kmedoids.shape[1]
print("image_f_s_c_with_kmedoids rows, cols: ", rows_c2,cols_c2)
print("image_f_s_c_with_kmedoids[0]", image_f_s_c_with_kmedoids[0])

image_f_s_c_with_kmedoids_clip = np.clip(image_f_s_c_with_kmedoids.astype('uint8'), 0, 255)
print("image_f_s_c_with_kmedoids_clip[0]", image_f_s_c_with_kmedoids_clip[0])

#Reshape image_f_s_c_with_kmedoids_clip to original (sampled) dimension
rows_c2_back_to_orig = 90
cols_c2_back_to_orig = 151
image_f_s_c_with_kmedoids_clip_final = image_f_s_c_with_kmedoids_clip.reshape(rows_c2_back_to_orig, cols_c2_back_to_orig, 3)
print("image_f_s_c_with_kmedoids_clip_final.shape", image_f_s_c_with_kmedoids_clip_final.shape)
print("image_f_s_c_with_kmedoids_clip_final.shape [0][1][2]", \
    image_f_s_c_with_kmedoids_clip_final.shape[0], image_f_s_c_with_kmedoids_clip_final.shape[1], image_f_s_c_with_kmedoids_clip_final.shape[2])
print(image_f_s_c_with_kmedoids_clip_final)

#Save and display output image
skimage.io.imsave('beach_s_c_with_kmedoids.png', image_f_s_c_with_kmedoids_clip_final)
io.imshow(image_f_s_c_with_kmedoids_clip_final)
io.show()