# Leonardo Cesar Cerqueira - 8937483
# SCC0251 - 2018/1Sem
# Trabalho 1 - Geracao de Imagens

import sys
import numpy as np
import imageio
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
            scene[x, y] = math.floor((x/param) - (math.sqrt(y/param)))
    return scene

# function 4
def simple_rand(scene, seed):
    for x in range(scene.shape[0]):
        for y in range(scene.shape[1]):
            scene[x,y] = random.random()
    return scene

# function 5
def random_walk(scene, seed):
    return scene

# Checking that all parameters were provided. Exits if they weren't.
if len(sys.argv) < 8:
    print("Not enough parameters were provided")
    sys.exit(0)

# Parsing input parameters
reference_img = sys.argv[1]
scene_dimension = int(sys.argv[2])
gen_function = int(sys.argv[3])
gen_parameter = float(sys.argv[4])
target_dimension = int(sys.argv[5])
target_quant = int(sys.argv[6])
rand_seed = int(sys.argv[7])

# Setting the seed
random.seed(rand_seed)

# Creating matrix of zeros of C x C dimesions
scene = np.zeros((scene_dimension, scene_dimension))

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
    simple_rand(scene, rand_seed)
else:
    print ("Invalid function parameter")
    sys.exit(0)

offset_scene = scene + math.fabs(np.min(scene))
formatted_scene = (offset_scene * 255 / np.max(offset_scene)).astype(np.uint8)
imageio.imwrite("output_images/out" + str(gen_function) + ".png", formatted_scene)