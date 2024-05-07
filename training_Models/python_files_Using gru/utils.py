import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# out look of the model
batch_size = 64
Number_of_images = 100
Image_size = (40, 800)
Image_embedding_size = 256
Image_length = 100
Text_embedding_size = 228
Max_Number_of_Words = 45


# Decoder parameters
Layers = 2

E_TO_D_Input_size = Image_embedding_size
E_TO_D_output_size = 64

D_Input_size = 64
D_output_size = 64

D_To_Output_Input_size = 64
D_To_Output_output_size = Text_embedding_size


# reverse Embedding parameters
Reverse_Input_size = Text_embedding_size
Reverse_output_size = Text_embedding_size 

drop_prob = 0.2