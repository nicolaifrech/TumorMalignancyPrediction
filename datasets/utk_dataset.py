import os
from PIL import Image
from torch.utils.data import Dataset

from datasets.splits import create_train_val_test_split

class UTKFaceDataset(Dataset):
    def __init__(self, data_folder, split_files=None, split='train', transform=None):
        """
        Args:
            data_folder: path to UTKFace image folder (e.g., 'utkface/UTKFace')
            split_files: dict with keys 'train', 'val', 'test' and file lists
            split: which subset to load ('train', 'val', or 'test')
            transform: optional transform to apply to images
        """
        self.data_folder = data_folder
        self.transform = transform

        if split_files is not None:
            self.image_files = split_files[split]
        else:
            self.image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]

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
