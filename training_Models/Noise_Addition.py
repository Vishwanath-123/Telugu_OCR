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
NoiseImagePath = '/home/ocr/teluguOCR/Dataset/Cropped_Dataset/Images'
NoiseLabelPath = '/home/ocr/teluguOCR/Dataset/Cropped_Dataset/Labels'

# Define the transformation to convert the images to tensors and perform any other necessary preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),  # Ensure PIL
    transforms.Grayscale(num_output_channels=1),  # Convert to single channel greyscale
    transforms.ToTensor()  # Convert to a PyTorch tensor
])


acchulu = ['అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ౠ', 'ఌ', 'ౡ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ', 'అం', 'అః']
hallulu = ['క', 'ఖ', 'గ', 'ఘ', 'ఙ',
           'చ', 'ఛ', 'జ', 'ఝ', 'ఞ',
           'ట', 'ఠ', 'డ', 'ఢ', 'ణ',
           'త', 'థ', 'ద', 'ధ', 'న',
           'ప', 'ఫ', 'బ', 'భ', 'మ',
           'య', 'ర', 'ల', 'వ', 'శ', 'ష', 'స', 'హ', 'ళ', 'క్ష', 'ఱ']
vallulu = ['ా', 'ి', 'ీ', 'ు' , 'ూ', 'ృ', 'ౄ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', 'ం', 'ః', 'ఁ', 'ౕ', 'ౖ', 'ౢ' ]
connector = ['్']
numbers = ['౦', '౧', '౨', '౩', '౪', '౫', '౬', '౭', '౮', '౯']
splcharacters= [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')',
              '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[',
              '\\', ']', '^', '_', '`', '{', '|', '}', '~', '1','2', '3', '4', '5', '6', '7', '8', '9', '0', 'ఽ']
spl = splcharacters + numbers

bases = acchulu + hallulu + spl
vms = vallulu
cms = hallulu

print("Bases: ", len(bases))
print("Vms: ", len(vms))
print("Cms: ", len(cms))

characters = bases+vms+cms+connector

base_mapping = {}
i = 2
for x in bases:
  base_mapping[x] = i
  i+=1

vm_mapping = {}
i = 2
for x in vms:
  vm_mapping[x] = i
  i+=1

cm_mapping = {}
i = 2
for x in cms:
  cm_mapping[x] = i
  i+=1

  
# creates a list of ductionaries with each dictionary reporesenting a term
def wordsDicts(s):
  List = []
  for i in range(len(s)):
    x = s[i]
    prev = ''
    if i > 0: prev = s[i-1]
    #----------------------------------is it a base term-----------------------
    if((x in acchulu or x in hallulu)  and prev != connector[0]):
      List.append({})
      List[-1]['base'] = x
    #----------------------------if it is a consonant modifier-----------------
    elif x in hallulu and prev == connector[0]:
      if(len(List) == 0):
        print(x)
      if('cm' not in List[-1]): List[-1]['cm'] = []
      List[len(List)-1]['cm'].append(x)

      #---------------------------if it is a vowel modifier--------------------
    elif x in vallulu:
      if(len(List) == 0):
        print(x)

      if('vm' not in List[-1]): List[-1]['vm'] = []
      List[len(List)-1]['vm'].append(x)

      #----------------------------it is a spl character-----------------------
    elif x in spl:
      List.append({})
      List[len(List)-1]['base'] = x
    else:
      continue
  return List


def index_encoding(s):
  List = wordsDicts(s)
  onehot = []
  for i in range(len(List)):
    D = List[i]
    onehotbase=  [1]
    onehotvm1 =  [1]
    onehotvm2 =  [1]
    onehotvm3 =  [1]
    onehotvm4 =  [1]
    onehotcm1 =  [1]
    onehotcm2 =  [1]
    onehotcm3 =  [1]
    onehotcm4 =  [1]


    onehotbase[0] = base_mapping[D['base']]

    it = 1
    if('vm' in D):
      for j in D['vm']:
        if it == 1:
          onehotvm1[0] = vm_mapping[j]
        elif it == 2:
          onehotvm2[0] = vm_mapping[j]
        elif it == 3:
          onehotvm3[0] = vm_mapping[j]
        elif it == 4:
          onehotvm4[0] = vm_mapping[j]
        it += 1
    
    it = 1
    if('cm' in D):
      for j in D['cm']:
        if it == 1:
          onehotcm1[0] = cm_mapping[j]
        elif it == 2:
          onehotcm2[0] = cm_mapping[j]
        elif it == 3:
          onehotcm3[0] = cm_mapping[j]
        elif it == 4:
          onehotcm4[0] = cm_mapping[j]
        it += 1
    onehoti = onehotbase + onehotvm1 + onehotvm2 + onehotvm3 + onehotvm4 + onehotcm1 + onehotcm2 + onehotcm3 + onehotcm4 #size 110 + 4*21 + 4*38 = 346
    onehot.append(onehoti)
  return torch.tensor(onehot)

def index_decoder(List):
  x = ""
  for onehoti in List:
    if onehoti[0] > 1:
      x += bases[onehoti[0]-2]

    if onehoti[5] > 1:
      x += connector[0]
      x += cms[onehoti[5]-2]
    if onehoti[6] > 1:
      x += connector[0]
      x += cms[onehoti[6]-2]
    if onehoti[7] > 1:
      x += connector[0]
      x += cms[onehoti[7]-2]
    if onehoti[8] > 1:
      x += connector[0]
      x += cms[onehoti[8]-2]

    if onehoti[1] > 1:
      x += vms[onehoti[1]-2]
    if onehoti[2] > 1:
      x += vms[onehoti[2]-2]
    if onehoti[3] > 1:
      x += vms[onehoti[3]-2]
    if onehoti[4] > 1:
      x += vms[onehoti[4]-2]
  return x



# Noise type 0
def add_gaussian_noise(img):
  img = transform(image)
  gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.1, clip=True))
  return gauss_img

# Noise type 1
inkbleed = InkBleed(intensity_range=(0.2, 1),
                    kernel_size=(5, 5),
                    severity=(0.2, 0.4)
                    )

# Noise type 2
doubleexposure = DoubleExposure(gaussian_kernel_range=(3,6),
                                offset_direction=1,
                                offset_range=(2,6),
                                )

# Noise type 3
hollow = Hollow(hollow_median_kernel_value_range = (51, 51),
                hollow_min_width_range=(1, 1),
                hollow_max_width_range=(20, 20),
                hollow_min_height_range=(1, 1),
                hollow_max_height_range=(20, 20),
                hollow_min_area_range=(10, 10),
                hollow_max_area_range=(500, 500),
                hollow_dilation_kernel_size_range = (3, 3),
                )

# Noise type 4
letterpress = Letterpress(n_samples=(20, 500),
                          n_clusters=(30, 80),
                          std_range=(150, 500),
                          value_range=(10, 15),
                          value_threshold_range=(128, 128),
                          blur=1
                          )

# Noise type 5
lighting_gradient_linear_static = LightingGradient(light_position=None,
                                              direction=45,
                                              max_brightness=255,
                                              min_brightness=0,
                                              mode="linear_static",
                                              linear_decay_rate = 0.5,
                                              transparency=0.5
                                              )

# Noise type 6
low_ink_periodic_line_consistent =  LowInkPeriodicLines(count_range=(1, 5),
                                                        period_range=(1, 3),
                                                        use_consistent_lines=True,
                                                        noise_probability=0.1,
                                                        )

# Noise type 7
shadowcast = ShadowCast(shadow_side = "bottom",
                        shadow_vertices_range = (2, 3),
                        shadow_width_range=(0.5, 0.8),
                        shadow_height_range=(0.5, 0.8),
                        shadow_color = (0, 0, 0),
                        shadow_opacity_range=(0.5,0.6),
                        shadow_iterations_range = (1,2),
                        shadow_blur_kernel_range = (51, 151),
                        )

# Noise type 8
folding = Folding(fold_count=2,
                  fold_noise=0.0,
                  fold_angle_range = (-180,180),
                  gradient_width=(0.05, 0.07),
                  gradient_height=(0.01, 0.05),
                  backdrop_color = (0,0,0),
                  )

# Noise type 9
book_binder_down = BookBinding(shadow_radius_range=(10, 10),
                              curve_range_right=(10, 10),
                              curve_range_left=(10, 10),
                              curve_ratio_right = (0.05, 0.05),
                              curve_ratio_left = (0.1, 0.1),
                              mirror_range=(0, 0),
                              binding_align = 0.5,
                              binding_pages = (10,10),
                              curling_direction=0,
                              backdrop_color=(255, 255, 255),
                              enable_shadow=0,
                              use_cache_images = 0,
                              )

# Noise type 10
inkshifter_obj = InkShifter(
    text_shift_scale_range=(5, 10),
    text_shift_factor_range=(1, 4),
    text_fade_range=(0, 2),
    noise_type = "random",
)


# load the txt file and read the file line by line.
def read_file_lines(filename):
    lines = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                lines.append(line.strip())  # Remove trailing newline characters
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return lines

lines = read_file_lines("/home/ocr/teluguOCR/Dataset/final_strings.txt")

num = 1
a = b = 0

types = []

for i in range(1, 189531+1):
    imagename = "Image" + str(i) + ".png"
    image = cv2.imread(os.path.join(dir, imagename))
    image = np.array(image)

    line = lines[i-1]

    random_subset = [random.randint(0, 11)]

    # Apply the selected noise functions to the image based on the random subset
    for noise_type in random_subset:
        if noise_type == 0:
            image = add_gaussian_noise(image)
        elif noise_type == 1:
            image = inkbleed(image)
            image = transform(image)
        elif noise_type == 2:
            image = doubleexposure(image)
            image = transform(image)
        elif noise_type == 3:
            image = hollow(image)
            image = transform(image)
        elif noise_type == 4:
            image = letterpress(image)
            image = transform(image)
        elif noise_type == 5:
            image = lighting_gradient_linear_static(image)
            image = transform(image)
        elif noise_type == 6:
            image = low_ink_periodic_line_consistent(image)
            image = transform(image)
        elif noise_type == 7:
            image = shadowcast(image)
            image = transform(image)
        elif noise_type == 8:
            image = folding(image)
            image = transform(image)
        elif noise_type == 9:
            image = book_binder_down(image)
            image = transform(image)
        elif noise_type == 10:
            image = inkshifter_obj(image)
            image = transform(image)
        else:
            image = transform(image)
            pass

    # convert the image tensor to PIL format
    transform2 = transforms.ToPILImage()
    image = transform2(image)
    image = image.convert('L')

    m = image.size[1]//40
    if m == 0 or image.size[0]//m == 0:
      continue
    image = image.resize((image.size[0]//m, 40))

    image = torch.tensor(np.array(image))

    image = image.to(dtype=torch.float64)

    if(image.shape[0] == 1):
        image = image[0]

      
    if line[0] not in bases:
        continue
    
    # considering only images with width less than 800
    if image.shape[1] > 800 or index_encoding(line).shape[0] > 45:
        continue
    
    
    image[image> 225] = 255
    image[image < 0] = 0

    image = 255 - image

    types.append(random_subset[0])

    label = index_encoding(line)

    save_path_im = f'{NoiseImagePath}/' + "Image" + str(num) + ".pt"
    save_path_lb = f'{NoiseLabelPath}/' + "Label" + str(num) + ".pt"
    num += 1
    torch.save(image,save_path_im)
    torch.save(label,save_path_lb)


    del image
    del random_subset
    del save_path_im, save_path_lb
    del imagename
    del noise_type
    
    print(i, end = '\r')    

print(num)
torch.save(types, '/home/ocr/teluguOCR/Dataset/Cropped_Dataset/NoiseTypes.pt')