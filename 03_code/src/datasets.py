import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.metrics import get_dynamic_weights

class MVTecDataset(Dataset):
    def __init__(self, root_path, category, transform=None, mask_transform=None):
        self.root = os.path.join(root_path, category)
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths, self.mask_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        img_paths, mask_paths, labels = [], [], []
        test_dir = os.path.join(self.root, 'test')
        gt_dir = os.path.join(self.root, 'ground_truth')
        if not os.path.exists(test_dir):
            return [], [], []

        for defect_type in os.listdir(test_dir):
            defect_dir = os.path.join(test_dir, defect_type)
            if not os.path.isdir(defect_dir):
                continue
            for img_name in os.listdir(defect_dir):
                img_paths.append(os.path.join(defect_dir, img_name))
                if defect_type == 'good':
                    labels.append(0)
                    mask_paths.append(None)
                else:
                    labels.append(1)
                    gt_name = img_name.split('.')[0] + '_mask.png'
                    mask_paths.append(os.path.join(gt_dir, defect_type, gt_name))
        return img_paths, mask_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_pil = Image.open(img_path).convert('RGB')
        weights = get_dynamic_weights(image_pil)

        if self.transform:
            image = self.transform(image_pil)
        mask_path = self.mask_paths[idx]
        if mask_path is None:
            mask = torch.zeros((1, 224, 224))
        else:
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = transforms.ToTensor()(mask)
        return image, mask, self.labels[idx], weights


class CastingDataset(Dataset):
    def __init__(self, root_path, transform=None, mask_transform=None):
        self.root = os.path.join(root_path, 'test') if os.path.exists(os.path.join(root_path, 'test')) else root_path
        self.transform = transform
        self.image_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        img_paths, labels = [], []
        ok_dir = os.path.join(self.root, 'ok_front')
        def_dir = os.path.join(self.root, 'def_front')
        if os.path.exists(ok_dir):
            for img in os.listdir(ok_dir):
                img_paths.append(os.path.join(ok_dir, img))
                labels.append(0)
        if os.path.exists(def_dir):
            for img in os.listdir(def_dir):
                img_paths.append(os.path.join(def_dir, img))
                labels.append(1)
        return img_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_pil = Image.open(self.image_paths[idx]).convert('RGB')
        weights = get_dynamic_weights(image_pil)

        if self.transform:
            image = self.transform(image_pil)
        mask = torch.zeros((1, 224, 224))
        return image, mask, self.labels[idx], weights


class MagneticTileDataset(Dataset):
    def __init__(self, root_path, transform=None, mask_transform=None):
        self.root = root_path
        self.transform = transform
        self.image_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        img_paths, labels = [], []
        folders = [f for f in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, f))]
        for folder in folders:
            folder_path = os.path.join(self.root, folder)
            is_normal = 'free' in folder.lower()
            img_dir = os.path.join(folder_path, 'Imgs')
            if os.path.exists(img_dir):
                for img_name in os.listdir(img_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_paths.append(os.path.join(img_dir, img_name))
                        labels.append(0 if is_normal else 1)
        return img_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_pil = Image.open(self.image_paths[idx]).convert('RGB')
        weights = get_dynamic_weights(image_pil)

        if self.transform:
            image = self.transform(image_pil)
        mask = torch.zeros((1, 224, 224))
        return image, mask, self.labels[idx], weights


class TestDataset(Dataset):
    """ Dataset loader for unlabelled test inference. """
    def __init__(self, root_path, transform=None):
        self.root = root_path
        self.transform = transform
        self.image_paths = []
        
        for root, _, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_pil = Image.open(img_path).convert('RGB')
        weights = get_dynamic_weights(image_pil)

        if self.transform:
            image = self.transform(image_pil)
            
        filename = os.path.relpath(img_path, self.root)
        return image, filename, weights