# Leonardo Cesar Cerqueira - 8937483
# SCC0251 - 2018/1Sem
# Trabalho 1 - Geracao de Imagens

import numpy as np
import math
import random

# function 1
def simple_sum(scene):
    for x in range(scene.shape[0]):
        for y in range(scene.shape[1]):
            scene[x, y] = x + y
    return scene

# function 2
def sine_sum(scene, param):
    for x in range(scene.shape[0]):
        for y in range(scene.shape[1]):
            scene[x, y] = math.fabs(math.sin(x/param) + math.sin(y/param))
    return scene

# function 3
def sqrt_sub(scene, param):
    for x in range(scene.shape[0]):
        for y in range(scene.shape[1]):
            scene[x, y] = x/param - math.sqrt(y/param)
    return scene

# function 4
def simple_rand(scene, seed):
    random.seed(rand_seed)
    for x in range(scene.shape[0]):
        for y in range(scene.shape[1]):
            scene[x,y] = random.random()
    return scene

# function 5
def random_walk(scene, seed):
    x = 0
    y = 0
    scene[x,y] = 1
    random.seed(rand_seed)
    for i in range(int(1 + (scene.shape[0] * scene.shape[0]) / 2)):
        dx = random.randint(-1, 1)
        x = (x + dx) % scene.shape[0]
        scene[x, y] = 1
        dy = random.randint(-1, 1)
        y = (y + dy) % scene.shape[0]
        scene[x, y] = 1
    return scene

# This function normalizes "data" between range min_t - max_t
def normalize_between(data, min_t, max_t):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = data * (max_t - min_t) + min_t
    return data


# Parsing input parameters
reference_img_file = str(input()).rstrip()
scene_dimension = int(str(input()).rstrip())
gen_function = int(str(input()).rstrip())
gen_parameter = float(str(input()).rstrip())
target_dimension = int(str(input()).rstrip())
target_quant = int(str(input()).rstrip())
rand_seed = int(str(input()).rstrip())

# Creating matrix of zeros of C x C dimesions
scene = np.zeros((scene_dimension, scene_dimension))
image = (np.zeros((target_dimension, target_dimension))).astype(np.uint8)

# Calling the function specified
if (gen_function == 1):
    simple_sum(scene)
elif (gen_function == 2):
    sine_sum(scene, gen_parameter)
elif (gen_function == 3):
    sqrt_sub(scene, gen_parameter)
elif (gen_function == 4):
    simple_rand(scene, rand_seed)
elif (gen_function == 5):
    random_walk(scene, rand_seed)
else:
    print ("Invalid function parameter")
    sys.exit(0)

# Normalizing and quantizing the generated image
norm_scene = normalize_between(scene, 0.0, float(2 ** 16 - 1))
formatted_scene = (normalize_between(norm_scene, 0.0, 255.0)).astype(np.uint8)

# Down scaling the image
ratio = (scene_dimension / target_dimension * 1.0)
for i in range(target_dimension):
    for j in range(target_dimension):
        i_left_lim = math.floor(i * ratio)
        i_right_lim = math.ceil(i * ratio + ratio - 1)
        j_left_lim = math.floor(j * ratio)
        j_right_lim = math.ceil(j * ratio + ratio - 1)

        image[i, j] = np.max(formatted_scene[i_left_lim : i_right_lim, j_left_lim : j_right_lim])

image = image >> (8 - target_quant)

# Loading the reference image
try:
    ref_image = (np.load(reference_img_file)).astype(np.uint8)
except:
    print("Something went wrong when loading the file")
    sys.exit(0)

float_image = image.astype(float)
float_ref_image = ref_image.astype(float)

# Taking the sum of the squared diff
diff = math.sqrt(np.sum((float_image - float_ref_image) ** 2))
print(diff)
