import skimage
import sklearn
import sklearn_extra
from skimage import io
import sklearn.cluster
import sklearn_extra.cluster 
import numpy as np

def initialize_K_centroids(image_arr, K):
    """ Choose K points from image_arr at random """
    random_selection = np.random.choice(len(image_arr), K, replace=False)
    #print("random_selection: ", random_selection)
    return image_arr[random_selection, :]

def find_closest_centroids(image_arr, centroids):
    len_image_arr = len(image_arr)
    closest_centroids_index_arr = np.zeros(len_image_arr)
    for index in range(len_image_arr):
        # Find distances to all of the K centroids; len(distances) = K
        distances = np.linalg.norm(image_arr[index, :] - centroids[:,:], axis=1)    #axis=1 => computation is across columns

        # Assign closest cluster to closest_centroids_index_arr[index]
        closest_centroids_index_arr[index] = np.argmin(distances)

    return closest_centroids_index_arr       #each element in 'closest_centroids_index_arr' will have a value between 0 and K-1
                                             #len(losest_centroids_index_arr) = len_image_arr


def compute_means(image_arr, closest_centroids_index_arr, K):
    n = image_arr.shape[1]   # n will be 3 for R,G,B
    centroids = np.zeros((K, n))     # will be initialized to [[0,0,0] .... [0,0,0]]
    for k in range(K):
        subset_image_arr = image_arr[np.where(closest_centroids_index_arr == k)]
        #mean = [np.mean(count) for count in subset_image_arr.T]
        #centroids[k] = mean
        i=0
        for count in subset_image_arr.T:
            centroids[k,i] = np.mean(count)
            i = i+1
    return centroids

def find_k_means(image_arr, K, max_iters=10):
    centroids = initialize_K_centroids(image_arr, K)
    previous_centroids = centroids
    for count in range(max_iters):
        closest_centroids_index_arr = find_closest_centroids(image_arr, centroids)
        centroids = compute_means(image_arr, closest_centroids_index_arr, K)
        if (centroids == previous_centroids).all():
            # The centroids aren't moving anymore.
            return centroids
        else:
            previous_centroids = centroids

    return centroids, closest_centroids_index_arr



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
print("image_f : Number of dimensions, rows, deep, total-size, length-of-1st-dim ", \
    image_f.ndim, image_f.shape[0], image_f.shape[1], image_f.size, len(image_f))
print("image_f[0]", image_f[0])
print("image_f[68479]", image_f[68479])  #214*320-1

centroid_colors, closest_centroid_colors_index_arr = find_k_means(image_f, 16, max_iters=10)
closest_centroid_colors_index_arr = find_closest_centroids(image_f, centroid_colors)

closest_centroid_colors_index_arr_int = closest_centroid_colors_index_arr.astype(int)

print("centroid_colors.dtype :", centroid_colors.dtype)
print("closest_centroid_colors_index_arr.dtype :", closest_centroid_colors_index_arr.dtype)
print("closest_centroid_colors_index_arr_int.dtype :", closest_centroid_colors_index_arr_int.dtype)
print("centroid_colors :", centroid_colors)
print("closest_centroid_colors_index_arr :", closest_centroid_colors_index_arr)
print("closest_centroid_colors_index_arr_int :", closest_centroid_colors_index_arr_int)

#Replace each pixel value with its nearby centroid and creating image_f_c_with_kmeans / 2D np array 
image_f_c_with_kmeans = centroid_colors[closest_centroid_colors_index_arr_int]


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
skimage.io.imsave('beach_c_without_kmeans_lib.png', image_f_c_with_kmeans_clip_final)
io.imshow(image_f_c_with_kmeans_clip_final)
io.show()

