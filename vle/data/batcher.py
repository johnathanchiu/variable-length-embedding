from webdataset.filters import pipelinefilter

import torchvision.transforms.functional as TF
import torch

from PIL import Image
import numpy as np
import random
import math


################################
######## FOR REFERENCE #########
################################
# pil_modes_mapping = {
#     InterpolationMode.NEAREST: 0,
#     InterpolationMode.BILINEAR: 2,
#     InterpolationMode.BICUBIC: 3,
#     InterpolationMode.NEAREST_EXACT: 0,
#     InterpolationMode.BOX: 4,
#     InterpolationMode.HAMMING: 5,
#     InterpolationMode.LANCZOS: 1,
# }
################################

INTERPOLATION_MODE = TF.InterpolationMode.BICUBIC


def ar_preserving_resize(
    image, new_height, new_width, interpolation=INTERPOLATION_MODE
):
    assert isinstance(image, Image.Image)

    width, height = image.size
    aspect_ratio = width / height
    resize_dim = (round(1 / aspect_ratio * new_width), new_width)
    if resize_dim[0] < new_height:
        resize_dim = (new_height, round(aspect_ratio * new_height))

    assert resize_dim[0] >= new_height and resize_dim[1] >= new_width

    image = TF.resize(image, size=resize_dim, interpolation=interpolation)
    image = TF.center_crop(image, output_size=(new_height, new_width))
    return image


class AspectRatiosBatcher(object):
    def __init__(
        self,
        resolution_constant,
        aspect_ratios,
        verbose=True,
    ):
        self.cache, self.cache_bins = self.setup_cache(
            aspect_ratios, resolution_constant
        )
        if verbose:
            print(f"AspectRatiosBatcher::cache_bins: {self.cache_bins}")

        self.batched = pipelinefilter(self._batched)

    def setup_cache(self, aspect_ratios, resolution_constant):
        cache_bins = []
        for aspect_ratio in aspect_ratios:
            # Compute width & height using resolution constant
            aspect_ratio = aspect_ratio[0] / aspect_ratio[1]
            new_width = int(round(math.sqrt(1 / aspect_ratio * resolution_constant)))
            new_height = int(round(resolution_constant / new_width))
            # Round width & height to nearest multiple of 64
            new_width = int(64 * round(new_width / 64.0))
            new_height = int(64 * round(new_height / 64.0))
            # Add them as cache bin keys
            cache_bins.append((new_height, new_width))
            if (new_height, new_width) != (new_width, new_height):
                cache_bins.append((new_width, new_height))

        # Setup cache
        cache = {}
        for bin_key in cache_bins:
            cache[bin_key] = []

        return cache, cache_bins

    def _batched(self, data, batchsize, collation_fn, partial):
        samples = iter(data)
        try:
            while True:
                ready_bin = self.get_next_ready_bin(size=batchsize)
                if ready_bin is None:
                    self.fill_cache(next(samples))
                else:
                    yield collation_fn(ready_bin)
        except StopIteration:
            pass

        # potentially yield partial bins and prefetched bins
        bin_keys = list(self.cache.keys())
        for bin_key in bin_keys:
            ready_bin = self.cache[bin_key]
            if len(ready_bin) > 0:
                n_batches = math.ceil(len(ready_bin) / batchsize)
                for i_batch in range(n_batches):
                    batch = ready_bin[i_batch * batchsize : (i_batch + 1) * batchsize]
                    if partial or len(batch) == batchsize:
                        yield collation_fn(batch)

    def get_image_transforms(self):
        return [self.resize_to_closest_bin]

    def resize_to_closest_bin(self, x):
        bin_key = self.find_bin_key(x)
        x = ar_preserving_resize(x, new_height=bin_key[0], new_width=bin_key[1])
        return x

    def fill_cache(self, x):
        bin_key = self.find_bin_key(x["jpg"])
        self.cache[bin_key].append(x)

    def find_bin_key(self, img):
        if isinstance(img, (torch.Tensor, np.ndarray)):
            assert len(img.shape) == 3 and img.shape[2] in [1, 3], img.shape
            height, width = img.shape[:2]
        elif isinstance(img, Image.Image):
            width, height = img.size
        else:
            raise TypeError(img)
        aspect_ratio = width / height
        comp_fn = lambda k: abs(k[1] / k[0] - aspect_ratio)
        return min(self.cache_bins, key=comp_fn)

    def clear_cache_at_bin(self, bin_key, n=None):
        assert bin_key in self.cache
        if n is not None:
            assert isinstance(n, int)
            del self.cache[bin_key][:n]
        else:
            self.cache[bin_key] = []

    def check_bin_cond(self, bin_length, size):
        if bin_length >= size:
            return True, size
        return False, size

    def get_next_ready_bin(self, size=1):
        bin_keys = list(self.cache.keys())
        random.shuffle(bin_keys)
        for bin_key in bin_keys:
            data = self.cache[bin_key]
            cond_flag, batch_size = self.check_bin_cond(len(data), size)
            if cond_flag:
                ret_data = data[:batch_size]
                self.clear_cache_at_bin(bin_key, batch_size)
                return ret_data
        return None
