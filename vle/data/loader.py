from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl

from .path_loaders import ImageNetDataset, CelebDataset, CIFAR10Dataset, INaturalistDataset


class CollectiveDataloader(pl.LightningDataModule):
    def __init__(self, datasets, num_workers=8, batch_size=10, shuffle=True):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.train_set = CollectiveDataset(datasets, "train")
        self.validation_set = CollectiveDataset(datasets, "valid")
        self.test_set = CollectiveDataset(datasets, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )


class CollectiveDataset(pl.LightningDataModule):
    dataset_loaders = {
        "celeb": CelebDataset,
        "imagenet": ImageNetDataset,
        "cifar": CIFAR10Dataset,
        "inaturalist": INaturalistDataset,
    }

    def __init__(self, datasets, split="train"):
        super().__init__()
        datasets_list = []
        for dataset, info in datasets.items():
            if dataset not in self.dataset_loaders:
                print("Skipping {dataset} dataset, could not be found")
                continue
            if split not in info:
                print(f"Skipping {split} split for {dataset}")
                continue
            datasets_list.append(
                self.dataset_loaders[dataset](info["path"], info[split])
            )
        self.datasets = ConcatDataset(datasets_list) if datasets_list else None

    def __len__(self):
        if self.datasets is None:
            return 0
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]
