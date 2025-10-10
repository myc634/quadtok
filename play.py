from diffusers import AutoencoderKL
import os
import logging

model = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae", local_files_only=True)