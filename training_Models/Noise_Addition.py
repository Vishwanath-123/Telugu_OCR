from torchvision import datasets
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image
from skimage.util import random_noise
from PIL import Image
from augraphy import *
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dir = '/home/ocr/teluguOCR/Dataset/Images/'
noisepath = '/home/ocr/teluguOCR/Dataset/Noised_Images'

# Define the transformation to convert the images to tensors and perform any other necessary preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),  # Ensure PIL
    transforms.ToTensor()  # Convert to a PyTorch tensor
])

# Function to add Gaussian noise to an image
def add_gaussian_noise(img):
  gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.01, clip=True))
  return gauss_img

# Function to add salt and pepper noise to an image
def add_salt_and_pepper_noise(img):
  s_and_p_img = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=0.1, clip=True))
  return s_and_p_img

# Function to add speckle noise to an image
def add_speckle_noise(img):
  speckle_noise_img = torch.tensor(random_noise(img, mode='speckle', mean=0, var=0.01, clip=True))
  return speckle_noise_img

# Function to add dirty drum roll noise to an image
add_dirty_drum_noise = DirtyDrum(ksize=(3,3))

# Function to add dirtyroller noise to an image
add_dirty_roller_noise = DirtyRollers()

# Function to add motion blur to an image
def add_motion_blur(img):
  kernel_size = 4
  angle = np.random.uniform(0, 45)
  kernel_v = np.zeros((kernel_size, kernel_size)) 
  kernel_h = np.copy(kernel_v) 
  kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
  kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
  kernel_v /= kernel_size 
  kernel_h /= kernel_size
  kernel = ((angle/90)*kernel_v + (1-(angle/90))*kernel_h)
  mb_image = cv2.filter2D(img, -1, kernel)
  return mb_image


count = 0
for i in range(1, 189531+1):
    imagename = "Image" + str(i) + ".png"
    image = cv2.imread(os.path.join(dir, imagename), cv2.IMREAD_GRAYSCALE)
    image = np.array(image)
    image = transform(image)

    random_subset = [random.randint(0, 2)]

    # Apply the selected noise functions to the image based on the random subset
    for noise_type in random_subset:
        if noise_type == 0:
            image = add_gaussian_noise(image)
            image = image*255
        elif noise_type == 1:
            image = add_dirty_roller_noise(cv2.imread(os.path.join(dir , imagename), cv2.IMREAD_GRAYSCALE))
            image = torch.tensor(image)
        elif noise_type == 2:
            image = add_motion_blur(cv2.imread(os.path.join(dir, imagename), cv2.IMREAD_GRAYSCALE))
            image = torch.tensor(image)
    
    image = image.to(dtype=torch.float64)

    if(image.shape[0] == 1):
        image = image[0]


    save_path = f'{noisepath}/' + "Image" + str(1000*count + i) + ".pt"
    torch.save(image,save_path)
    del image
    del random_subset
    del save_path
    del imagename
    del noise_type
    print(i, end = '\r')    