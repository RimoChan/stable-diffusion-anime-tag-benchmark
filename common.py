import os
import re
from typing import Union, Optional

import requests
from PIL import Image
import rimo_storage.cache
from transformers import AutoProcessor, AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

import ml_danbooru


服务器地址 = f'http://127.0.0.1:7860'


def ml_danbooru标签(image_list: list[Union[str, bytes, os.PathLike]]) -> dict[str, dict[str, float]]:
    超d = {}
    for image in image_list:
        tags = ml_danbooru.get_tags_from_image(Image.open(image), threshold=0.5, keep_ratio=True)
        超d[image] = tags
    return 超d


def safe_name(s: str):
    return re.sub(r'[\\/:*?"<>|]', lambda m: str(ord(m.group()[0])), s)


def 上网(p, j=None, method='get'):
    r = getattr(requests, method)(p, json=j)
    r.reason = r.text[:4096]
    r.raise_for_status()
    return r.json()


@rimo_storage.cache.disk_cache(serialize='pickle')
def 缓存上网(p, j=None, method='get'):
    return 上网(p, j=j, method=method)


def check_model(要测的模型: list[tuple[Optional[str]]]):
    所有模型 = [i['model_name'] for i in 上网(f'{服务器地址}/sdapi/v1/sd-models')]
    所有VAE = [i['model_name'] for i in 上网(f'{服务器地址}/sdapi/v1/sd-vae')]
    assert set([i[0] for i in 要测的模型]) < set(所有模型), f'模型只能从{所有模型}中选择'
    assert set([i[1] for i in 要测的模型]) < set(所有VAE + [None]), f'VAE只能从{所有VAE}或None中选择'


def cos(a, b):
    return a.dot(b).item() / (a.norm() * b.norm()).item()


clip = None
clip_processor = None
def 图像相似度(img1, img2):
    global clip, clip_processor
    if clip is None:
        clip = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32')
        clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
    inputs = clip_processor(images=[img1, img2], return_tensors="pt")
    outputs = clip(**inputs)
    return cos(outputs.image_embeds[0], outputs.image_embeds[1])


clip_text = None
clip_tokenizer = None
def 图文相似度(img, text: str):
    global clip, clip_processor, clip_text, clip_tokenizer
    if clip is None:
        clip = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32')
        clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
    if clip_text is None:
        clip_text = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    outputs_1 = clip(**clip_processor(images=[img], return_tensors="pt"))
    outputs_2 = clip_text(**clip_tokenizer([text], padding=True, return_tensors="pt"))
    return cos(outputs_1.image_embeds[0], outputs_2.text_embeds[0])
