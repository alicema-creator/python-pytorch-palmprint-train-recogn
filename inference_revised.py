
# # 查询图片路径
# query_image_path = './waiting_recognition_photo/00015.bmp'
# # 文件夹路径
# folder_path = './ROI/session1'

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models import MyDataset
from models import compnet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\ndevice-> ', device, '\n\n')


# Load test dataset
class CustomDataset(Dataset):
    def __init__(self, txt, transforms=None, train=False):
        self.samples = []
        with open(txt, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                self.samples.append((img_path, int(label)))
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transforms:
            image = self.transforms(image)
        return image, label


test_set = './data/test.txt'
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match input size of the model
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (adjust based on your model's preprocessing)
])

testset = CustomDataset(txt=test_set, transforms=transform, train=False)
batch_size = 1
data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)



# Print the best match result
print('\nBest match found:')
print('Sample index: %d, Matching distance: %.2f, Label: %d, Image path: %s' %
      (best_match_index, best_match_distance, best_match_label, best_match_image_path))