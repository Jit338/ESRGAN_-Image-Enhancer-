import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from IPython.display import Image as IPythonImage, display

class SRDataset(Dataset):
    def __init__(self, hr_dir, patch_size=128, scale=4):
        super().__init__()
        self.image_paths = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.ToTensor()
        ])

        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(patch_size // scale, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        hr_image = Image.open(self.image_paths[idx]).convert("RGB")
        
        hr_tensor = self.hr_transform(hr_image)

        lr_tensor = self.lr_transform(hr_tensor) 

        return lr_tensor, hr_tensor