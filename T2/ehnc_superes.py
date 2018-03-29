# Leonardo Cesar Cerqueira - 8937483
# SCC0251 - 2018/1Sem
# Trabalho 2 - Realce e Superresolução

import numpy as np
import math
import imageio

# This function calculates the cumulative histogram for the image given and
# returns it as a numpy array of type int
def cumulative_histogram(image):
    hist = np.zeros(256).astype(int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i, j]] += 1
    
    return hist

# This function returns an equalized version of the 'image', using the histogram
# 'hist', and the total of pixels 'pixel_count' accounted for in the histogram
def transfer_function(image, hist, pixel_count):
    eq_img = (255.0 / pixel_count) * hist[image]
    
    return eq_img

# This function returns the gamma adjusted version of 'image', using the 'gamma' parameter
def gamma_adjustment(image, gamma):
    adjt_img = np.floor(255 * ((image / 255.0) ** 1/gamma))

    return adjt_img

# This function returns the super-res image built from the four low-res provided
def superresolution(image1, image2, image3, image4):
    super_res = np.zeros((image1.shape[0] * 2, image1.shape[1] * 2))
    for i in range(super_res.shape[0]):
        for j in range(super_res.shape[1]):
            if (i % 2 == 0 and j % 2 == 0):
                super_res[i, j] = image1[i // 2, j // 2]
            elif (i % 2 == 0 and j % 2 == 1):
                super_res[i, j] = image3[i // 2, j // 2]
            elif (i % 2 == 1 and j % 2 == 0):
                super_res[i, j] = image2[i // 2, j // 2]
            elif (i % 2 == 1 and j % 2 == 1):
                super_res[i, j] = image4[i // 2, j // 2]

    return super_res

# This function returns the RMSE error between two images
def rmse_error(image1, image2):
    resolution = image1.shape[0] * image1.shape[1]
    squared_diff = (image1 - image2) ** 2
    error = math.sqrt(np.sum(squared_diff) / resolution)

    return error

# Reading input
lowres_base_name = str(input()).rstrip()
highres_name = str(input()).rstrip()
enh_method = int(input())
enh_paramenter = float(input())

# Loading images
lowres1 = imageio.imread(lowres_base_name + "1.png")
lowres2 = imageio.imread(lowres_base_name + "2.png")
lowres3 = imageio.imread(lowres_base_name + "3.png")
lowres4 = imageio.imread(lowres_base_name + "4.png")
ref_highres = imageio.imread(highres_name + ".png")

# If either 1 or 2, histograms will be used for the enhancement
if (enh_method == 1 or enh_method == 2):
    # Calculte the histogram for each image
    hist1 = cumulative_histogram(lowres1)
    hist2 = cumulative_histogram(lowres2)
    hist3 = cumulative_histogram(lowres3)
    hist4 = cumulative_histogram(lowres4)

    # If using method 1, equalize each image using their individual histogram
    if (enh_method == 1):
        lowres1 = transfer_function(lowres1, hist1, lowres1.shape[0] * lowres1.shape[1])
        lowres2 = transfer_function(lowres2, hist2, lowres2.shape[0] * lowres2.shape[1])
        lowres3 = transfer_function(lowres3, hist3, lowres3.shape[0] * lowres3.shape[1])
        lowres4 = transfer_function(lowres4, hist4, lowres4.shape[0] * lowres4.shape[1])
    # If using method 2, equalize each image using the accumulated histogram for all four
    else:
        # The accumulated histogram is the sum of all the others
        hist_all = hist1 + hist2 + hist3 + hist4
        lowres1 = transfer_function(lowres1, hist_all, 4 * lowres1.shape[0] * lowres1.shape[1])
        lowres2 = transfer_function(lowres2, hist_all, 4 * lowres2.shape[0] * lowres2.shape[1])
        lowres3 = transfer_function(lowres3, hist_all, 4 * lowres3.shape[0] * lowres3.shape[1])
        lowres4 = transfer_function(lowres4, hist_all, 4 * lowres4.shape[0] * lowres4.shape[1])

# If using method 3, enhance using gamma adjustment
elif (enh_method == 3):
    lowres1 = gamma_adjustment(lowres1, enh_paramenter)
    lowres2 = gamma_adjustment(lowres2, enh_paramenter)
    lowres3 = gamma_adjustment(lowres3, enh_paramenter)
    lowres4 = gamma_adjustment(lowres4, enh_paramenter)

# If using method 0,  no enhancing is done

# Bulding the super-res image
highres = superresolution(lowres1, lowres2, lowres3, lowres4)

# Calculating the difference between highres and the reference image
diff = rmse_error(ref_highres, highres)

# Printing the difference factor
print("%.4f" % diff)