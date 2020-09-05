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
    return image_arr[random_selection, :]

def min_distances_array(image_arr, centroids):           #this returns distances from all points to their respective centroids
    len_image_arr = len(image_arr)
    len_centroids = len(centroids)
    min_distances_arr = np.zeros((len_image_arr)) 
    #print("distances_arr.shape :", distances_arr.shape)
    for index in range(len_image_arr):
        # Find distances to all of the K centroids; distances_arr is len_image_arr x len_centroids
        distances = np.linalg.norm(image_arr[index] - centroids, axis=1)    #axis=1 => computation is across columns
        #'distances' represents distance of image_arr[index] point to all the centroids => it is an 1-d array of len_centroids
        #broadcasting property of numpy is being used here https://numpy.org/doc/stable/user/basics.broadcasting.html
        #print("distance:", distance)
        min_distances_arr[index] = np.min(distances)
    return min_distances_arr       


def find_closest_centroids(image_arr, centroids):
    len_image_arr = len(image_arr)
    closest_centroids_index_arr = np.zeros(len_image_arr) #closest_centroids_index_arr is an 1-d array
    for index in range(len_image_arr):
        # Find distances to all of the K centroids; len(distances) = K
        distances = np.linalg.norm(image_arr[index, :] - centroids[:,:], axis=1)    #axis=1 => computation is across columns
        #'distance' represents distance of image_arr[index] to all the centroids => it is an 1-d array of len_centroids
        #broadcasting property of numpy is being used here https://numpy.org/doc/stable/user/basics.broadcasting.html
        # Assign closest cluster to closest_centroids_index_arr[index]
        closest_centroids_index_arr[index] = np.argmin(distances)   #https://numpy.org/doc/stable/reference/generated/numpy.argmin.html

    return closest_centroids_index_arr       #each element in 'closest_centroids_index_arr' will have a value between 0 and K-1
                                             #len(closest_centroids_index_arr) = len_image_arr


def compute_new_centroids(image_arr, closest_centroids_index_arr, centroids_orig):
    n = image_arr.shape[1]   # n will be 3 for R,G,B
    K = len(centroids_orig) 
    print("Entered compute_new_centroids")
    sum_min_distances = np.sum(min_distances_array(image_arr, centroids_orig))
    centroids = np.copy(centroids_orig)
    print("In compute_new_centroids function: Now will go through k in range(K)")
    for k in range(K):
        print("centroid k=", k)
        centroids_temp = np.copy(centroids)
        subset_image_arr = image_arr[np.where(closest_centroids_index_arr == k)]
        print("initial centroids for k = ", k, centroids)
        
        for i in range(len(subset_image_arr)):
            print("centroid k, point i in that cluster, len(subset_image_arr:", k,i, len(subset_image_arr))
            centroids_temp[k] = np.copy(subset_image_arr[i])
            sum_min_distances_temp = np.sum(min_distances_array(image_arr, centroids_temp))
            print("k, i, sum_min_distances, sum_min_distances_temp", k,i, sum_min_distances, sum_min_distances_temp)
            if sum_min_distances_temp < sum_min_distances:
                print("k, i centroids[k] getting ipdated")
                centroids[k] = np.copy(centroids_temp[k])  #https://docs.scipy.org/doc/numpy-1.9.2/reference/generated/numpy.copy.html
                sum_min_distances = sum_min_distances_temp
                print("updated centroids for k, i = ", k, i, centroids)
            if (i == (len(subset_image_arr) - 1)):
                print("centroids for k, i = ", k, i, centroids)
        
        print("final updated centroids for k = ", k, centroids)
    
    print("updated centroids being returned from compute_new_centroids function ", centroids)
    return centroids

def find_k_centroids(image_arr, K, max_iters=10):
    centroids = initialize_K_centroids(image_arr, K)
    print("initial centroids :", centroids)
    previous_centroids = np.copy(centroids)   # https://docs.scipy.org/doc/numpy-1.9.2/reference/generated/numpy.copy.html
    for count in range(max_iters):
        print("Entering iteration : ", count)
        closest_centroids_index_arr = find_closest_centroids(image_arr, centroids)
        print("Now calling compute_new_centroids" )
        centroids = compute_new_centroids(image_arr, closest_centroids_index_arr, centroids)
        print("centroids.shape, closest_centroids_index_arr.shape :", centroids.shape, closest_centroids_index_arr.shape)
        print("closest_centroids_index_arr :", closest_centroids_index_arr)
        print("previous_centroids :", previous_centroids)
        print("centroids :", centroids)
        if (centroids == previous_centroids).all():
            print("The centroids aren't moving anymore ")
            return centroids, closest_centroids_index_arr
        else:
            previous_centroids = np.copy(centroids)  # https://docs.scipy.org/doc/numpy-1.9.2/reference/generated/numpy.copy.html
    
    print("reached limit of max number of iterations :", count)
    print("Final centroids :", centroids)
    print("Final closest_centroids_index_arrcentroids :", closest_centroids_index_arr)
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

#create a sample np array that samples every 5th pixel
image_f_s = np.empty([30375, 3])   #68480/2 ~ 30375
#print("Empty image_f_s: ", image_f_s)
for i in range(30375):
    image_f_s[i] = image_f[i*2]
print("Sampled image_f_s: ", image_f_s)
print("image_f_s: Number of dimensions, rows, deep, total-size, length-of-1st-dim ", image_f_s.ndim, image_f_s.shape[0], image_f_s.shape[1], \
    image_f_s.size, len(image_f_s))



centroid_colors, closest_centroid_colors_index_arr = find_k_centroids(image_f_s, 16, max_iters=1)
closest_centroid_colors_index_arr = find_closest_centroids(image_f_s, centroid_colors)

closest_centroid_colors_index_arr_int = closest_centroid_colors_index_arr.astype(int)

print("centroid_colors.dtype :", centroid_colors.dtype)
print("closest_centroid_colors_index_arr.dtype :", closest_centroid_colors_index_arr.dtype)
print("closest_centroid_colors_index_arr_int.dtype :", closest_centroid_colors_index_arr_int.dtype)
print("centroid_colors :", centroid_colors)
print("closest_centroid_colors_index_arr :", closest_centroid_colors_index_arr)
print("closest_centroid_colors_index_arr_int :", closest_centroid_colors_index_arr_int)

#Replace each pixel value with its nearby centroid and creating image_f_c_with_kmeans / 2D np array 
image_f_s_c_with_kmedoids = centroid_colors[closest_centroid_colors_index_arr_int]


print("image_f_s_c_with_kmedoids shape: ", image_f_s_c_with_kmedoids.shape)
rows_c1 = image_f_s_c_with_kmedoids.shape[0]
cols_c1 = image_f_s_c_with_kmedoids.shape[1]
print("image_f_s_c_with_kmedoids rows, cols: ", rows_c1,cols_c1)
print("image_f_s_c_with_kmedoids[0]", image_f_s_c_with_kmedoids[0])


image_f_s_c_with_kmedoids_clip = np.clip(image_f_s_c_with_kmedoids.astype('uint8'), 0, 255)
print("image_f_s_c_with_kmedoids_clip[0]", image_f_s_c_with_kmedoids_clip[0])


#Reshape image to original dimension 135 x 225 = 30375
rows_c2_back_to_orig = 135
cols_c2_back_to_orig = 225
image_f_s_c_with_kmedoids_clip_final = image_f_s_c_with_kmedoids_clip.reshape(rows_c2_back_to_orig, cols_c2_back_to_orig, 3)
print("image_f_s_c_with_kmedoids_clip_final.shape", image_f_s_c_with_kmedoids_clip_final.shape)
print("image_f_s_c_with_kmedoids_clip_final.shape [0][1][2]", \
    image_f_s_c_with_kmedoids_clip_final.shape[0], image_f_s_c_with_kmedoids_clip_final.shape[1], image_f_s_c_with_kmedoids_clip_final.shape[2])
print(image_f_s_c_with_kmedoids_clip_final)


#Save and display output image
skimage.io.imsave('beach_c_without_kmedoids_lib-1.png', image_f_s_c_with_kmedoids_clip_final)
io.imshow(image_f_s_c_with_kmedoids_clip_final)
io.show()

