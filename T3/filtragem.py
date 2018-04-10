# Leonardo Cesar Cerqueira - 8937483
# SCC0251 - 2018/1Sem
# Trabalho 3 - Filtragem 1D

import numpy as np
import imageio

# This function reads the arbitrary filter from the keyboard and returns it
def manual_filter_gen(size):
    manual_filter = str(input()).rstrip().split(' ')
    manual_filter = [float(i) for i in manual_filter]

    return np.array(manual_filter)[0 : size]

# This functions calculates the gaussian filter using the size and standard deviation
# values provided and returns it
def gauss_filter_calc(size, std_dev):
    x = np.arange(-(size // 2), size // 2 + 1)
    gauss_filter = (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp((-1/(2 * (std_dev ** 2))) * np.power(x, 2.0))
    gauss_filter = gauss_filter / np.sum(gauss_filter)

    return gauss_filter

# This function does the convolution in the space domain
def space_domain(image1d, used_filter):
    filtered_image = np.zeros(image1d.shape[0])
    for i in np.arange(image1d.shape[0]):
        j = 0
        for k in np.arange(-(len(used_filter) // 2), len(used_filter) // 2 + 1):
            pos = (i + k) % image1d.shape[0] if i + k >= 0 else image1d.shape[0] + k
            filtered_image[i] += image1d[pos] * used_filter[j]
            j += 1

    return filtered_image

# Discrete Fourier Transform - Code taken from the course github repository
def DFT1D(A):
    
    F = np.zeros(A.shape, dtype=np.complex64)
    n = A.shape[0]

    x = np.arange(n)
    for u in np.arange(n):
        F[u] = np.sum(A*np.exp( (-1j * 2 * np.pi * u*x) / n ))

    return F

# Inverse Discrete Fourier Transform
def IDFT1D(A):
    
    Inv_F = np.zeros(A.shape, dtype=np.complex64)
    n = A.shape[0]

    x = np.arange(n)
    for u in np.arange(n):
        Inv_F[u] = np.sum(A*np.exp( (1j * 2 * np.pi * u*x) / n )) / n

    return Inv_F

# This function applies the Convolution Theory to filter the image using Fourier transforms
def frequency_domain(image1d, used_filter):
    if (len(used_filter) < image1d.shape[0]):
        ext_filter = np.concatenate((used_filter, np.zeros(image1d.shape[0] - len(used_filter))))

    filtered_image = IDFT1D(np.multiply(DFT1D(ext_filter), DFT1D(image1d)))

    return np.real(filtered_image)

 # This function normalizes "data" between range min_t - max_t
def normalize_between(data, min_t, max_t):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = data * (max_t - min_t) + min_t
    
    return data

# This function returns the RMSE error between two images
def rmse_error(image1, image2):
    resolution = image1.shape[0] * image1.shape[1]
    squared_diff = np.power((image1 - image2), 2.0)
    error = np.sqrt(np.sum(squared_diff) / resolution)

    return error

# Reading input and calculating the filter
used_filter = []
std_dev = 0
input_image_name = str(input()).rstrip()
filter_option = int(input())
filter_size = int(input())

if (filter_option == 1):
    used_filter = manual_filter_gen(filter_size)
elif (filter_option == 2):
    std_dev = float(input())
    used_filter = gauss_filter_calc(filter_size, std_dev)

filter_domain = int(input())

# Reading the image into memory
input_image = imageio.imread(input_image_name)

# Concatenating the image into an array
image_array = input_image.flatten().astype(float)

if (filter_domain == 1):
    filtered_image = space_domain(image_array, used_filter)
elif (filter_domain == 2):
    filtered_image = frequency_domain(image_array, used_filter)

# Reshaping and normalizing the filtered image
filtered_image2d = np.reshape(filtered_image, (-1, input_image.shape[1]))
filtered_image2d = normalize_between(filtered_image2d, 0.0, 255.0).astype(np.uint8)

# Calculating the difference between highres and the reference image
diff = rmse_error(input_image.astype(float), filtered_image2d.astype(float))

# Printing the difference factor
print("%.4f" % diff)