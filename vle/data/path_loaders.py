from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import os


class ImageNetDataset(Dataset):
    transform = transforms.Compose(
        [
            transforms.CenterCrop((384, 448)),
            transforms.ToTensor(),
        ]
    )

    def __init__(self, path, split):
        super().__init__()
        if split == "train":
            path = os.path.join(path, "ILSVRC2012_train")
        elif split == "valid":
            path = os.path.join(path, "ILSVRC2012_validation")
        else:
            raise AssertionError(f"Split {split} not available")
        self.dataset = datasets.ImageFolder(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset.__getitem__(idx)
        img = self.transform(img) * 2.0 - 1.0
        return img
    

class CIFAR10Dataset(Dataset):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    def __init__(self, path, split="train"):
        self.dataset = torchvision.datasets.CIFAR10(path, split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset.__getitem__(idx)
        img = self.transform(img)
        img = img * 2.0 - 1.0
        return img
        

class CelebDataset(Dataset):
    transform = transforms.Compose(
        [
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor(),
        ]
    )

    def __init__(self, path, split):
        super().__init__()
        self.dataset = torchvision.datasets.CelebA(
            path, split=split
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset.__getitem__(idx)
        img = self.transform(img) * 2.0 - 1.0
        return img


class INaturalistDataset(Dataset):
    transform = transforms.Compose(
        [
            transforms.CenterCrop((512, 512)),
            transforms.ToTensor(),   
        ]
    )

    def __init__(self, path, split):
        self.dataset = torchvision.datasets.INaturalist(
            path, version=split, transform=self.transform, 
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset.__getitem__(idx)
        img = img * 2.0 - 1.0
        return img

    