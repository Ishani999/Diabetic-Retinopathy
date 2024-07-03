import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import os
from scipy import misc
from PIL import Image
from skimage import exposure
from sklearn import svm

import scipy
from math import sqrt,pi
from numpy import exp
from matplotlib import pyplot as plt
import numpy as np
import glob
import matplotlib.pyplot as pltss
import cv2
from matplotlib import cm
import pandas as pd
from math import pi, sqrt
import pywt

import os
import cv2
import numpy as np



# Define the transformation with normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # These are ImageNet normalization values
                         std=[0.229, 0.224, 0.225])
])

# Define your filter kernel functions and related code
#L affects the size of the kernel 
def _filter_kernel_mf_fdog(L, sigma, t=3, mf=True):
    dim_y = int(L)
    dim_x = 2 * int(t * sigma)
    arr = np.zeros((dim_y, dim_x), 'f')

    ctr_x = dim_x / 2 
    ctr_y = int(dim_y / 2.)

    # Setting elements of the array to their x coordinate
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        arr[it.multi_index] = it.multi_index[1] - ctr_x
        it.iternext()

    two_sigma_sq = 2 * sigma * sigma
    sqrt_w_pi_sigma = 1. / (sqrt(2 * np.pi) * sigma)
    if not mf:
        sqrt_w_pi_sigma = sqrt_w_pi_sigma / sigma ** 2

    def k_fun(x):
        return sqrt_w_pi_sigma * np.exp(-x * x / two_sigma_sq)

    def k_fun_derivative(x):
        return -x * sqrt_w_pi_sigma * np.exp(-x * x / two_sigma_sq)

    if mf:
        kernel = k_fun(arr)
        kernel = kernel - kernel.mean()
    else:
        kernel = k_fun_derivative(arr)

    return cv2.flip(kernel, -1)

def gaussian_matched_filter_kernel(L, sigma, t=3):
    return _filter_kernel_mf_fdog(L, sigma, t, True)

def createMatchedFilterBank(K, n=12):
    rotate = 180 / n
    center = (K.shape[1] / 2, K.shape[0] / 2)
    cur_rot = 0
    kernels = [K]

    for i in range(1, n):
        cur_rot += rotate
        r_mat = cv2.getRotationMatrix2D(center, cur_rot, 1)
        k = cv2.warpAffine(K, r_mat, (K.shape[1], K.shape[0]))
        kernels.append(k)

    return kernels

def applyFilters(im, kernels):
    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)

# Function to process images in a directory with subdirectories for each class
def process_images_in_directory_with_classes(directory, output_directory, kernels):
    class_names = sorted(os.listdir(directory))  # Get sorted list of class folders
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            output_class_dir = os.path.join(output_directory, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(class_dir, filename)
                    output_image_path = os.path.join(output_class_dir, filename)
                    
                    image = cv2.imread(image_path)  # Read image in grayscale
                    processed_image = applyFilters(image, kernels)
                    
                    # Save processed image
                    cv2.imwrite(output_image_path, processed_image)




# Define the directories for the dataset
train_dir = 'diabetic-retinopathy/ml_model/dataset/train'
output_train_dir = 'diabetic-retinopathy/ml_model/processed_dataset/train'  # Output directory for processed images
test_dir = 'diabetic-retinopathy/ml_model/dataset/test'
output_test_dir = 'diabetic-retinopathy/ml_model/processed_dataset/test'

# Generate filter bank from matched filter kernel
gf = gaussian_matched_filter_kernel(10, 2)
bank_gf = createMatchedFilterBank(gf, 12)

# Process images in train directory with multiple classes
process_images_in_directory_with_classes(train_dir, output_train_dir, bank_gf)
process_images_in_directory_with_classes(test_dir, output_test_dir, bank_gf)

#print("Processing complete. Processed images saved to:", output_train_dir)