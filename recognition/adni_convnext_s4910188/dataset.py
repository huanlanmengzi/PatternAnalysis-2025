import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ADNIDataset(Dataset):
    """
    Load ADNI dataset from folder structure:
    AD_NC/train/AD/
    AD_NC/train/NC/
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._make_dataset()
        print(f" Loaded {len(self.samples)} image samples from {root_dir}")

    def _make_dataset(self):
        samples = []
        classes = {"AD": 1, "NC": 0}
        for label_name, label in classes.items():
            class_dir = os.path.join(self.root_dir, label_name)
            if not os.path.exists(class_dir):
                print(f" Missing folder: {class_dir}")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    path = os.path.join(class_dir, fname)
                    samples.append((path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
