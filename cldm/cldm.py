import os
from PIL import Image
import numpy as np

import einops
import torch
import torch.nn as nn

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img

from torchvision import models
import cv2

class OCTLDM(LatentDiffusion):

    def __init__(self, CF_key, num_global_feature, *args, **kwargs):
    # def __init__(self, CF_key, num_global_feature, num_local_feature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CF_key = CF_key

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        
        layer = -3
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:layer])
        self.feature_extractor = self.feature_extractor.to(self.device)

        self.global_process = torch.nn.Conv2d(1024, num_global_feature, kernel_size=1)
        # self.local_process = torch.nn.Linear(512, num_local_feature)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, _ = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        cond_global = batch[self.CF_key]
        # cond_global, cond_local = batch[self.CF_key]
        if bs is not None:
            cond_global = cond_global[:bs]
            # cond_local = cond_local[:bs]

        cond_global = cond_global.to(self.device)
        cond_global = einops.rearrange(cond_global, 'b h w c -> b c h w')
        cond_global_origin = cond_global
        cond_global = self.feature_extractor(cond_global)
        cond_global = self.global_process(cond_global)
        cond_global = einops.rearrange(cond_global, 'b c h w -> b (h w) c')
        cond_global = cond_global.to(memory_format=torch.contiguous_format).float()
        
        # 6 是六个方向，dd 是局部所取区域的宽度，长度 w 是 256
        # bs = cond_local.shape[0]
        # cond_local = cond_local.to(self.device)
        # cond_local = einops.rearrange(cond_local, 'b h w c -> b c h w')
        # cond_local_origin = cond_local
        # cond_local = einops.rearrange(cond_local, 'b c (d n) w -> (b c d n) w', n=6)
        # cond_local = self.local_process(cond_local)
        # cond_local = einops.rearrange(cond_local, '(b c d n) w -> b (c d n) w', b=bs, n=6, c=3)
        # cond_local = cond_local.to(memory_format=torch.contiguous_format).float()

        # cond_global torch.Size([20, 1024, 256]) 
        # cond_local torch.Size([20, 2304=64*2*6*3, 256])
        return x, dict(cond_global=[cond_global], cond_global_origin=[cond_global_origin])
        # return x, dict(cond_global=[cond_global], cond_local=[cond_local], cond_global_origin=[cond_global_origin], cond_local_origin=[cond_local_origin])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_global = torch.cat(cond['cond_global'], 1)
        # cond_local = torch.cat(cond['cond_local'], 1)
        context = dict(cond_global=cond_global)
        # context = dict(cond_global=cond_global, cond_local=cond_local)
        
        eps = diffusion_model(x=x_noisy, timesteps=t, context=context)

        return eps

    # @torch.no_grad()
    # def get_unconditional_conditioning(self, N):
    #     return self.get_learned_conditioning(["chaotic, shadow, worst quality, low quality"] * N)

    @torch.no_grad()
    def log_valid_images(self, batch, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)

        N = z.shape[0]
        log["CF_path"] = batch["CF_path"][:N] 
        cond_global = c["cond_global"][0][:N]
        # cond_local = c["cond_local"][0][:N]

        c_full = dict(cond_global=[cond_global])
        # c_full = dict(cond_global=[cond_global], cond_local=[cond_local])
        samples_cfg, _ = self.sample_log(cond=c_full,
                                            batch_size=N, ddim=use_ddim,
                                            ddim_steps=ddim_steps, eta=ddim_eta,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=None,
                                            )
        x_samples_cfg = self.decode_first_stage(samples_cfg)
        log["samples"] = x_samples_cfg

        return log

    def shared_step_test(self, batch, batch_idx):
        ddim_steps = 5
        images = self.log_valid_images(batch, ddim_steps=ddim_steps)
        foldername = f"./log_valid/1124900-ddim{ddim_steps}/samples"
        os.makedirs(foldername, exist_ok=True)

        for k in ["samples"]:
            images[k] = images[k][0:]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                images[k] = torch.clamp(images[k], -1., 1.)

        for k in ["samples"]:
            bs = images[k].shape[0]
            for image_idx in range(bs): 
                image = images[k][image_idx] # [6, 512, 512]
                image = (image + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                image = image.transpose(0, 1).transpose(1, 2).squeeze(-1)
                image = image.numpy()
                image = (image * 255).astype(np.uint8)
                for i in range(6):
                    CF_name = images['CF_path'][image_idx].split('.')[0]
                    item_path = os.path.join(foldername, CF_name)
                    os.makedirs(item_path, exist_ok=True)
                    path = os.path.join(foldername, CF_name, f"{CF_name}_{i}.jpg")
                    img = image[:,:,i]
                    # 缩放这个黑白图像，高度固定在320像素，宽度固定在448像素
                    img = cv2.resize(img, (448, 320))
                    Image.fromarray(img).save(path)
           
    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        cond_global = c["cond_global"][0][:N]
        # cond_local = c["cond_local"][0][:N]
        cond_global_origin = c["cond_global_origin"][0][:N]
        # cond_local_origin = c["cond_local_origin"][0][:N]
        
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control_global"] = cond_global_origin * 2.0 - 1.0
        # log["control_local"] = cond_local_origin * 2.0 - 1.0

        if unconditional_guidance_scale > 1.0:
            c_full = dict(cond_global=[cond_global])
            # c_full = dict(cond_global=[cond_global], cond_local=[cond_local])
            samples_cfg, _ = self.sample_log(cond=c_full,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=None,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log["samples"] = x_samples_cfg

        return log
