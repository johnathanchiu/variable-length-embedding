from vle.utils import instantiate_from_config, load_config
from vle.data.loader import CollectiveDataloader

from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

from torchmetrics import StructuralSimilarityIndexMeasure
import skimage.measure as measure
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np
from tqdm import tqdm
import cv2

torch.set_grad_enabled(False)

MODELS = {
    "vanilla": {
        "config": "/fsx/home/johnathan/variable-length-embeddings/logs/2023-05-08T15-22-54_vanilla_vle/vanilla_vle_laion.yaml",
        "ckpt": "/fsx/home/johnathan/variable-length-embeddings/logs/2023-05-08T15-22-54_vanilla_vle/checkpoints/last.ckpt",
    },
}


class VQVAE(nn.Module):
    repo_name = "CompVis/stable-diffusion-v1-4"
    model_name = "stabilityai/sd-vae-ft-mse"
    def __init__(self):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(self.model_name)
        sd_model = StableDiffusionPipeline.from_pretrained(
            self.repo_name, vae=vae
        )
        self.model = sd_model.vae
        
    def forward(self, x, n_tokens, log=False, **kwargs):
        if log:
            tokens = []
        
        epsilon = 1e-4
        rec = torch.zeros_like(x)
        for _ in range(n_tokens):
            err = x - rec
            i_rec = self.model(err).sample
            rec = rec + i_rec
            
            if log:
                tokens.append(i_rec)
            
        if log:
            intermediates = {
                "model_output": tokens,
            }
            return rec, intermediates

        return rec

@torch.no_grad()
def eval_metrics(model, data, n_tokens, device, num_samples=200):
    eval_mses = []
    eval_ssims = []
    img_entropies = []
    
    total_samples = 0
    for i, batch in enumerate(tqdm(data)):
        inputs = batch.cuda(device)
        
        *_, intermediates = model(inputs, n_tokens, log=True, postprocess=False)
    
        # Add final reconstruction to the intermediates list
        intermediates = intermediates["model_output"]

        # Compute MSE & SSIM metrics
        batch_mses = []
        batch_ssims = []
        rec = torch.zeros_like(inputs)
        for idx, i_rec in enumerate(intermediates):
            rec = rec + i_rec
            clamped_rec = rec.clamp(-1.0, 1.0)
            intermediate_mse = F.mse_loss(inputs, clamped_rec).item()
            intermediate_ssim = ssim_loss(inputs, clamped_rec).item()
            batch_mses.append(intermediate_mse)
            batch_ssims.append(intermediate_ssim)
        eval_mses.append(batch_mses)
        eval_ssims.append(batch_ssims)
        
        # Compute entropy of the images
        batch_img_entropies = []
        for img in inputs:
            img = img.moveaxis(0, -1).cpu().numpy() 
            img = (img + 1.0) / 2.0 * 255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            im_entropy = measure.shannon_entropy(img)
            batch_img_entropies.append(im_entropy)
        img_entropies.append(batch_img_entropies)
        
        total_samples += inputs.size(0)
        if total_samples >= num_samples:
            break
            
    assert len(eval_mses) == len(eval_ssims) == len(img_entropies)
    num_batches = len(eval_mses)
    final_mse_metric = sum([bm[-1] for bm in eval_mses]) / num_batches
    final_ssim_metric = sum([bm[-1] for bm in eval_ssims]) / num_batches
            
    return {
        "final_mse": final_mse_metric,
        "final_ssim": final_ssim_metric,
        "intermediate_mse": np.array(eval_mses),
        "intermediate_ssim": np.array(eval_ssims),
        "entropy": np.array(img_entropies),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae", default=False, action="store_true")
    parser.add_argument("--model_name", default="vanilla", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--max_tokens", default=12, type=int)
    parser.add_argument("--num_samples", default=500, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    args = parser.parse_args() 

    torch.manual_seed(args.seed)


    ssim_loss = StructuralSimilarityIndexMeasure(
        data_range=1.0, kernel_size=7
    ).cuda(args.device)

    data = CollectiveDataloader(
        {
            "inaturalist": {
                "path": "/fsx/home/johnathan/data",
                "valid": "2021_valid",
            },
        },
        batch_size=5,
        shuffle=False,
        num_workers=12,
    ).val_dataloader()
    
    if args.vqvae:
        model, model_name = VQVAE(), "vqvae"
    else:
        model_config = MODELS[args.model_name]["config"]
        model_ckpt = MODELS[args.model_name]["ckpt"]
        config = load_config(model_config)
        model = instantiate_from_config(config["model"])
        model.load_weights(model_ckpt, verbose=True)
        model_name = args.model_name
    model = model.cuda(args.device).eval()

    print(f"Maximum number of tokens: {args.max_tokens}")
    print(f"Number of samples: {args.num_samples}")

    metrics_dict = eval_metrics(model, data, args.max_tokens, args.device, num_samples=args.num_samples)

    np.save(
        f"{model_name}_tokens{args.max_tokens}_samples{args.num_samples}_seed{args.seed}", 
        metrics_dict,
    )

