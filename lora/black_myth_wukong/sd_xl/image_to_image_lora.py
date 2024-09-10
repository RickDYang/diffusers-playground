import os

import random
import numpy as np
import torch

from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image


def load_base_pipeline(base_model_name: str):
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16
    ).to("cuda")
    pipeline.enable_model_cpu_offload()
    return pipeline


def load_lora(pipeline, lora_model_name, lora_rank):
    pipeline.load_lora_weights(
        lora_model_name, weight_name=f"pytorch_lora_weights_{lora_rank}.safetensors"
    )
    return pipeline


def is_nsfw(image):
    img = np.array(image)
    return np.all(img == 0)


def infer_one(pipeline, prompt: str, negative_prompt: str, source_image_path: str):
    i = 0
    source_image = load_image(source_image_path)
    image = None
    while i < 10:
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_image,
        ).images[0]

        if is_nsfw(image):
            print(f"NSFW image generated for image: {source_image_path}")
        else:
            break
    return image


def infer(
    pipeline, lora_rank: int, source_folder: str, prompt: str, negative_prompt: str
):
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")) and (
                "-lora-" not in file
            ):
                file_path = os.path.join(root, file)
                filename_no_ext, _ = os.path.splitext(file)
                image = infer_one(
                    pipeline=pipeline,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    source_image_path=file_path,
                )

                image.save(
                    f"{source_folder}/{filename_no_ext}-lora-{lora_rank:02d}.png"
                )


def image_to_image(
    base_model_name: str,
    lora_model_name: str,
    lora_ranks: list[int],
    source_folder: str,
    prompt: str,
    negative_prompt: str,
):
    r"""
    Generate images from texts with prompts combinations via a base stable diffuser model
    and series lora models.
    The generated images are saved to the source directory , with the name pattern
    "{file_name}-lora-{rank:02d}.png".

    Parameters:
        base_model_name (str): The name of the base stable diffuser model.
        lora_mode_name (str): The name of lora model, which may contrains weights file of various ranks, which name pattern is
            "pytorch_lora_weights_{rank}.safetensors".
        lora_ranks (list[int]): The ranks of lora models.
        source_folder (str): The folder containing the source images to generate.
        prompt (str): The positive prompt containing target style prompt and good prompt instruction.
        negative_prompt (str): The negative prompt to prevent generating bad images.
    """
    pipeline = load_base_pipeline(base_model_name)
    for lora_rank in lora_ranks:
        pipeline = load_lora(pipeline, lora_model_name, lora_rank)
        infer(
            pipeline=pipeline,
            lora_rank=lora_rank,
            source_folder=source_folder,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )


if __name__ == "__main__":
    random.seed(1234)
    torch.manual_seed(1234)
    image_to_image(
        "stabilityai/stable-diffusion-xl-base-1.0",
        "rickdyang/sd-xl-lora-blackmyth",
        [4],
        "./in",
        "(black myth wukong style), (masterpiece), highly detailed, highest quality, best quality, ultra high resolution, highres",
        "nsfw, magic circle, BadDream, UnrealisticDream, worst quality, poor details, low quality, jpeg artifacts",
    )
