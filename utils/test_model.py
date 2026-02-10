from src.s2v_model import WanModel_S2V
from diffusers import ConfigMixin
import json
from omegaconf import OmegaConf
import torch
from src.models.vae import Wan2_1_VAE

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

config_path = "./configs/model_small.yaml"
config = OmegaConf.to_container(OmegaConf.load(config_path))

# model = WanModel_S2V(**config["model"])

# model = WanModel_S2V.from_pretrained(
#     "./checkpoints/model_small",
#     # config="./checkpoints/model_small/config.json",
#     # low_cpu_mem_usage=False,
#     # ignore_mismatched_sizes=True,
#     dtype=torch.bfloat16,
# )
# original_parameters = count_parameters(model)

# print(f"Original parameters: {original_parameters}")

# # model.save_pretrained("./checkpoints/model_small")

#   Test VAE
vae = Wan2_1_VAE(
    vae_pth="./checkpoints/s2v_model/wan_vae.pth",
    dtype=torch.bfloat16,
    device="cuda",
)

videos = [torch.randn(3, 33, 1024, 1024).to("cuda")]
zs = vae.encode(videos)
breakpoint()
print(zs[0].shape)