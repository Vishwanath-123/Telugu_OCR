import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# out look of the model
Number_of_images = 100
Image_size = (30, 800)
Image_embedding_size = 50
Image_length = 100
Text_embedding_size = 346
Max_Number_of_Words = 45


# LSTM parameters for the RNN
LSTM_Input_size = Image_embedding_size #50
LSTM_hidden_size = Text_embedding_size #346
LSTM_num_layers = 1
LSTM_output_size = LSTM_hidden_size #346

# reverse Embedding parameters
Reverse_Input_size = LSTM_output_size #374
Reverse_output_size = Text_embedding_size #374

drop_prob = 0.3