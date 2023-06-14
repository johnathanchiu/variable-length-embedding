from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class ImageLogger(Callback):
    def __init__(
        self,
        log_directory,
        batch_frequency,
        sample_latent=False,
        sample_intermediates=False,
        downsample_factor=0.25,
    ):
        super().__init__()
        self.batch_frequency = batch_frequency
        self.sample_latent = sample_latent
        self.sample_intermediates = sample_intermediates
        self.make_image_logging_dir(log_directory)
        self.downsample_factor = downsample_factor

    @rank_zero_only
    def make_image_logging_dir(self, log_directory):
        self.log_dir = f"{log_directory}/images"
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    @rank_zero_only
    def log_img(self, pl_module, batch, epoch_num, step, split="train"):
        batch = batch[:min(3, batch.size(0))]
        imgs_log = pl_module.log_img(
            batch,
            None,
            sample_intermediates=self.sample_intermediates,
            sample_latent=self.sample_latent,
        )
        reconstructions = self.stack_and_resize_imgs(imgs_log["reconstructions"], self.downsample_factor)
        original_imgs = batch.moveaxis(1, -1).detach().cpu().numpy()
        original_imgs = (original_imgs + 1.0) / 2.0 * 255
        original_imgs = self.stack_and_resize_imgs([[img] for img in original_imgs], self.downsample_factor)
        imgs = np.concatenate((original_imgs, reconstructions), axis=0)
        img_path = os.path.join(self.log_dir, f"epoch_{epoch_num:05}_step_{step:06}_reconstruction_{split}.jpg")
        self.save_img(imgs, img_path)

        del imgs_log["reconstructions"]

        for key in imgs_log:
            path = os.path.join(
                self.log_dir,
                f"epoch_{epoch_num:05}_step_{step:06}_{key}_{split}.jpg",
            )
            imgs = self.stack_and_resize_imgs(imgs_log[key], self.downsample_factor)
            self.save_img(imgs, path)

    def stack_and_resize_imgs(self, img_list, downsample_factor):
        vstacked_list = []
        for batch in img_list:
            resized_batch = []
            for img in batch:
                resized_batch.append(self.resize_img(img, downsample_factor))
            vstacked_list.append(np.vstack(resized_batch))
        return np.hstack(vstacked_list)

    def resize_img(self, img, downsample_factor):
        return cv2.resize(img, (0, 0), fx=downsample_factor, fy=downsample_factor)

    def save_img(self, img, path):
        if len(img.shape) == 3:
            img = img[..., ::-1]
        cv2.imwrite(path, img)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.batch_frequency == 0:
            self.log_img(
                pl_module,
                batch,
                trainer.current_epoch,
                trainer.global_step,
                split="train",
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (trainer.global_step + 1) % self.batch_frequency == 0:
            self.log_img(
                pl_module,
                batch,
                trainer.current_epoch,
                trainer.global_step,
                split="validation",
            )
