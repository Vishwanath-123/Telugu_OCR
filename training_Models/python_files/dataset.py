from torch.utils.data import Dataset, DataLoader
import torch
import os

class TeluguOCRDataset(Dataset):
    def __init__(self, image_file_path, label_file_path):
        self.image_file_path = image_file_path
        self.label_file_path = label_file_path
    
    def __len__(self):
        return len(os.listdir(self.image_file_path))

    def __getitem__(self, index):
        image = torch.load(os.path.join(self.image_file_path, "Image"+str(index+1) + '.pt'))
        label = torch.load(os.path.join(self.label_file_path, "Label"+str(index+1) + '.pt'))

        new_image = torch.zeros((1, 40, 800))
        new_label = torch.zeros((32, 9))

        new_image[0, :, :image.shape[1]] = image
        new_label[:label.shape[0], :] = label

        return new_image, new_label, label.shape[0]