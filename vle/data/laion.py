from vle.utils import instantiate_from_config

from torch.utils.data import DataLoader
import webdataset as wds

from einops import rearrange
import pytorch_lightning as pl
import torchvision
import torch
import os


def img_collation_fn(img_key):
    def collate(samples):
        img_samples = []
        for sample in samples:
            img_samples.append(sample[img_key])
        img_samples = torch.stack(img_samples)
        return img_samples.moveaxis(-1, 1)
    return collate


class WebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        tar_base,
        batch_size,
        partitions,
        num_workers=4,
        multinode=True,
        min_size=None,
        max_pwatermark=1.0,
        custom_batcher=None,
        persistent_workers=True,
    ):
        super().__init__()
        # batching functionality
        self.custom_batcher = None 
        if custom_batcher is not None:
            self.custom_batcher = instantiate_from_config(custom_batcher)
        self.batch_size = batch_size

        # dataset paritions
        self.train = partitions["train"] if "train" in partitions else None
        self.validation = partitions["validation"] if "validation" in partitions else None
        self.test = partitions["test"] if "test" in partitions else None

        # dataset loading 
        self.tar_base = tar_base
        self.multinode = multinode
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        
        # filter variables
        self.min_size = min_size  
        self.max_pwatermark = max_pwatermark 

    def make_loader(self, dataset_config):
        if "image_transforms" in dataset_config:
            assert self.custom_batcher is None
            image_transforms = [
                instantiate_from_config(tt) for tt in dataset_config["image_transforms"]
            ]
        else:
            assert self.custom_batcher is not None
            image_transforms = self.custom_batcher.get_image_transforms()

        image_transforms.extend(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")
                ),
            ]
        )
        image_transforms = torchvision.transforms.Compose(image_transforms)

        img_key = dataset_config.get("image_key", "jpeg")
        transform_dict = {img_key: image_transforms}

        shuffle = dataset_config.get("shuffle", 0)
        shardshuffle = shuffle > 0

        nodesplitter = (
            wds.shardlists.split_by_node
            if self.multinode
            else wds.shardlists.single_node_only
        )

        tars = os.path.join(self.tar_base, dataset_config["shards"])

        dset = (
            wds.WebDataset(
                tars,
                nodesplitter=nodesplitter,
                shardshuffle=shardshuffle,
                handler=wds.ignore_and_continue,
            )
            .repeat()
            .shuffle(shuffle)
        )
        print(f"Loading webdataset with {len(dset.pipeline[0].urls)} shards.")

        dset = (
            dset.select(self.filter_keys)
            .decode("pil", handler=wds.ignore_and_continue)
            .select(self.filter_image_aspects)
            .map_dict(**transform_dict, handler=wds.warn_and_continue)
        )

        if self.custom_batcher is not None:
            return dset.compose(
                self.custom_batcher.batched(
                    self.batch_size, partial=False, collation_fn=img_collation_fn(img_key),
                )
            )

        return dset.batched(
            self.batch_size, partial=False, collation_fn=img_collation_fn(img_key),
        )

    def filter_image_aspects(self, x):
        try:
            valid = True
            if self.min_size is not None and self.min_size > 1:
                enough_width = x["json"]["original_width"] >= self.min_size
                enough_height = x["json"]["original_height"] >= self.min_size
                valid = valid and enough_width and enough_height
            if self.max_pwatermark is not None and self.max_pwatermark < 1.0:
                return valid and x["json"]["pwatermark"] <= self.max_pwatermark
            return valid
        except Exception:
            return False

    def filter_keys(self, x):
        try:
            return "jpg" in x
        except Exception:
            return False

    def train_dataset(self):
        return self.make_loader(self.train)

    # def val_dataset(self):
    #     return self.make_loader(self.validation)

    # def test_dataset(self):
    #     return self.make_loader(self.test)

    def train_dataloader(self):
        dataset = self.make_loader(self.train)
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    # def val_dataloader(self):
    #     dataset = self.make_loader(self.validation)
    #     return DataLoader(
    #         dataset,
    #         batch_size=None,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         persistent_workers=self.persistent_workers,
    #     )

    # def test_dataloader(self):
    #     dataset = self.make_loader(self.test)
    #     return DataLoader(
    #         dataset,
    #         batch_size=None,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         persistent_workers=self.persistent_workers,
    #     )
