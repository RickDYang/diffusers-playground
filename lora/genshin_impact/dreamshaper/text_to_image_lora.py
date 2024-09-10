import os

from typing import Tuple
import json
import random

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from compel import Compel


def load_base_pipeline(base_model_name):
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_name, torch_dtype=torch.float16
    ).to("cuda")
    pipeline.enable_model_cpu_offload()
    # use compel to avoid the prompt being too long(>77 tokens)
    # but it will truncate long prompts
    compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    return pipeline, compel


def load_lora(pipeline, lora_model_name, rank):
    pipeline.load_lora_weights(
        lora_model_name,
        weight_name=f"pytorch_lora_weights_{rank}.safetensors",
        force_download=FORCE_DOWNLOAD,
    )
    return pipeline


# check the image is nsfw or not
# if all pixels are black, then it's nsfw
def is_nsfw(image):
    img = np.array(image)
    return np.all(img == 0)


def random_select(combination_prompts: list[list[str]]):
    res = []
    for sub_prompts in combination_prompts:
        res.append(random.choice(sub_prompts))
    return res


def try_infer(
    pipeline, compel, prompt: Tuple[str], combination_prompts: list[list[str]]
):
    selected_prompts = random_select(combination_prompts)
    positive_prompt = f"{', '.join(selected_prompts)}, {prompt[0]}"
    conditioning = compel.build_conditioning_tensor(positive_prompt)
    neg_conditioning = compel.build_conditioning_tensor(prompt[1])
    image = pipeline(
        prompt_embeds=conditioning,
        negative_prompt_embeds=neg_conditioning,
        width=WIDTH,
        height=HEIGHT,
        guidance_scale=GUIDANCE_SCALE,
    ).images[0]

    return image, selected_prompts


def infer(
    pipeline,
    compel,
    rank: int,
    prompts_pairs: list[Tuple[str, str]],
    combination_prompts: list[list[str]],
):
    total = len(prompts_pairs)
    for i, prompt in enumerate(prompts_pairs):
        j = 0
        image = None
        while j < 10:
            image, selected_prompts = try_infer(
                pipeline, compel, prompt, combination_prompts
            )
            if is_nsfw(image):
                print(f"NSFW image generated for prompts: {selected_prompts}")
            else:
                break
            j += 1

        filename = f"r{rank:02d}-{i:04d}"
        image.save(f"{OUTPUT_DIR}/{filename}.png")
        print(f"Generate {filename} done/{total}")


def load_prompts():
    with open("prompts.json", "r", encoding="utf-8") as file:
        data = json.load(file)
        return (
            data["major"],
            data["combinations"],
            data["positives"],
            data["negatives"],
        )


def texts_to_images(
    base_model_name: str,
    lora_model_name: str,
    lora_ranks: list[int],
    major_promts: list[str],
    combination_prompts: list[list[str]],
    positive_prompts: list[str],
    negative_prompts: list[str],
    num_images: int = 10,
):
    r"""
    Generate images from texts with prompts combinations via a base stable diffuser model
    and series lora models.
    The generated images are saved to the directory "./test/", with the name pattern
    "r{rank:02d}-{index}.png".

    Parameters:
        base_model_name (str): The name of the base stable diffuser model.
        lora_model (str): The name of lora model, which may contrains weights file of various ranks, which name pattern is
            "pytorch_lora_weights_{rank}.safetensors".
        lora_ranks (list[int]): The ranks of lora models.
        major_promts (list[str]): The major prompts, e.g. ["(genshin impact style)", "portrait"].
        combination_prompts (list[list[str]]): The combination prompts to randomly selected from for each image.
        positive_prompts (list[str]): The positive prompts
        negative_prompts (list[str]): The negative prompts
        num_images (int): The number of images to generate.
    """
    prompts_pairs = [
        (", ".join(major_promts + positive_prompts), ", ".join(negative_prompts))
    ] * num_images
    pipeline, compel = load_base_pipeline(base_model_name)
    # infer with the base model without lora
    # we set the rank to 0 to indicate the base model
    infer(pipeline, compel, 0, prompts_pairs, combination_prompts)
    for rank in lora_ranks:
        load_lora(pipeline, lora_model_name, rank)
        infer(pipeline, compel, rank, prompts_pairs, combination_prompts)


GUIDANCE_SCALE = 8
WIDTH = 384
HEIGHT = 512
# The huggingface API load_lora_weights have bugs
# which cannot download the weights file except the first weight safetenors file
# So we set FORCE_DOWNLOAD to True to force download all the weights files
FORCE_DOWNLOAD = True
OUTPUT_DIR = "./out"


if __name__ == "__main__":
    random.seed(1234)
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    prompts = load_prompts()
    texts_to_images(
        "xyn-ai/DreamShaper",
        "RickDYang/dreamshaper-lora-genshin",
        [4, 8, 16],
        *prompts,
        num_images=10,
    )
