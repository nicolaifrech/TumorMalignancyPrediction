import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class UTKFaceDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        age = int(filename.split('_')[0])  # Extract age from filename
        image_path = os.path.join(self.data_folder, filename)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, age 
