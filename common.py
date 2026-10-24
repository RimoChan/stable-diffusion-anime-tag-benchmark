import csv
import sys
import os
import re
import copy
import json
from pathlib import Path
from functools import lru_cache
from typing import Union, Optional, Iterable

import cv2
import torch
import orjson
import requests
import numpy as np
from PIL import Image
import rimo_storage.cache
from huggingface_hub import hf_hub_download
from timm.models import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoProcessor, AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, pipeline

import torch
import diffusers

# assert torch.__version__ == '2.11.0+cu130'
assert diffusers.__version__ == '0.40.0'


此处 = Path(__file__).parent
服务器地址 = f'http://127.0.0.1:7860'


def ml_danbooru标签(image_list: list[Union[str, bytes, os.PathLike]]) -> dict[str, dict[str, float]]:
    import ml_danbooru
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


def txt2img_nai3(p: dict) -> list[bytes]:
    import io
    import time
    import zipfile
    token = '...'
    api = "https://image.novelai.net/ai/generate-image"
    headers = {
        "authorization": f"Bearer {token}",
        "referer": "https://novelai.net",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
    }

    参数 = {}
    j = {
        "model": "nai-diffusion-3",
        "action": "generate",
        "parameters": 参数
    }
    j['input'] = p.pop('prompt')

    参数['width'] = p.pop('width')
    参数['height'] = p.pop('height')
    参数['steps'] = p.pop('steps')
    参数['negative_prompt'] = p.pop('negative_prompt')

    参数['scale'] = p.pop('cfg_scale')

    assert p.pop('sampler_name') == 'DPM++ 2M'
    assert p.pop('scheduler') == 'Karras'
    参数['sampler'] = 'k_dpmpp_2m'
    参数['scheduler'] = 'karras'

    p.pop('override_settings', None)

    参数['n_samples'] = 1    # n_samples传1、steps传28就不消耗Anlas，所以循环1下
    参数['steps'] = 28
    n_samples = p.pop('batch_size') * p.pop('n_iter')  
    seed = p.pop('seed')

    assert not p, f'剩下参数{p}不知道怎么转换……'

    ret = []
    for d_seed in range(n_samples):
        j['parameters']['seed'] = seed + d_seed
        for retry_i in range(60):
            try:
                r = requests.post(api, json=j, headers=headers)
                r.reason = r.text[:4096]
                r.raise_for_status()
            except Exception as e:
                print(repr(e))
                time.sleep(retry_i*10)
            else:
                break
        else:
            raise Exception('坏了！')
        with zipfile.ZipFile(io.BytesIO(r.content), mode="r") as z:
            img, = z.filelist
            ret.append(z.open(img).read())
    return ret


def check_model(要测的模型: list[tuple[str, Optional[str], str]]):
    所有模型 = [i['model_name'] for i in 上网(f'{服务器地址}/sdapi/v1/sd-models')]
    所有VAE = [i['model_name'] for i in 上网(f'{服务器地址}/sdapi/v1/sd-vae')]
    assert set([i[0] for i in 要测的模型]) < set(所有模型 + ['nai-diffusion-3']), f'模型只能从{所有模型}中选择，{set([i[0] for i in 要测的模型])-set(所有模型)}不行！'
    assert set([i[1] for i in 要测的模型]) < set(所有VAE + [None]), f'VAE只能从{所有VAE}或None中选择'


def cos(a, b):
    return a.dot(b).item() / (a.norm() * b.norm()).item()


clip = None
clip_processor = None
def 图像相似度(img1, img2):
    global clip, clip_processor
    if clip is None:
        clip = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32').to("cuda")
        clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
    inputs = clip_processor(images=[img1, img2], return_tensors="pt").to("cuda")
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


def 参数相同(a: dict, b: dict):     # 其实结果是不一样的，但是以前测试的时候忘了指定这个参数
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    a['override_settings'].pop('CLIP_stop_at_last_layers', None)
    a.pop('model_type', None)
    b['override_settings'].pop('CLIP_stop_at_last_layers', None)
    b.pop('model_type', None)
    if a['override_settings'].get('sd_vae'):
        a['override_settings']['sd_vae'] = a['override_settings']['sd_vae'].replace('.pt', '.safetensors').replace('.ckpt', '.safetensors')
    if b['override_settings'].get('sd_vae'):
        b['override_settings']['sd_vae'] = b['override_settings']['sd_vae'].replace('.pt', '.safetensors').replace('.ckpt', '.safetensors')
    a_sampler = (a.pop('sampler_index', None) or (' '.join([a.pop('sampler_name', ''), a.pop('scheduler', '')]))).lower()
    b_sampler = (b.pop('sampler_index', None) or (' '.join([b.pop('sampler_name', ''), b.pop('scheduler', '')]))).lower()
    return a == b and a_sampler == b_sampler


质量pipeline = None
def 图像质量(images) -> list[list[dict]]:
    global 质量pipeline
    if 质量pipeline is None:
        质量pipeline = pipeline("image-classification", model="shadowlilac/aesthetic-shadow", device=0)
    return 质量pipeline(images=images)


yolo_world_model = None
def yolo_world(图像名: str, 类名: Iterable) -> tuple:
    from yolo_world_onnx import YOLOWORLD
    global yolo_world_model
    if yolo_world_model is None:
        yolo_world_model = YOLOWORLD("R:/yolov8x-worldv2.onnx", device="0")
    yolo_world_model.set_classes([*类名])
    image = cv2.imdecode(np.fromfile(图像名, dtype=np.uint8), -1)
    boxes, scores, class_ids = yolo_world_model(image, conf=0.01)
    names = [yolo_world_model.names[i] for i in class_ids]
    return names, scores


@lru_cache()
def _Kaloscope_2_model():
    sys.path.append('S:/lsnet-test')
    from model import lsnet_artist

    device = 'cuda'
    checkpoint_path = hf_hub_download(repo_id='heathcliff01/Kaloscope2.0', filename='448-90.13/best_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['model']
    state = {k.removeprefix('module.'): v for k, v in state.items()}

    num_classes, feature_dim = state["head.l.weight"].shape
    model = create_model('lsnet_xl_artist_448', pretrained=False, num_classes=num_classes, feature_dim=feature_dim)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    data_config = resolve_data_config({"input_size": (3, 448, 448)}, model=model)
    transform = create_transform(**data_config, is_training=False)

    csv_path = hf_hub_download(repo_id='heathcliff01/Kaloscope2.0', filename='class_mapping.csv')
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        画师映射 = {int(row['class_id']): row['class_name'] for row in csv.DictReader(f)}
    return model, 画师映射, transform


def 识别画师(img: Image.Image, top_k: int = 5) -> list[tuple[str, float]]:
    model, 画师映射, 预处理 = _Kaloscope_2_model()
    input_tensor = 预处理(img.convert('RGB')).unsqueeze(0).to('cuda')
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    probs, indices = torch.topk(probabilities, top_k)
    return [(画师映射.get(i, str(i)), float(p)) for p, i in zip(probs.tolist(), indices.tolist())]


with open('./data/标签top1500.json', 'r') as f:
    标签top1500 = json.load(f)
    assert len(标签top1500) == 1500
with open('./data/标签top3000.json', 'r') as f:
    标签top3000 = json.load(f)
    assert len(标签top3000) == 3000

要测的标签 = 标签top1500



人频率1 = {k: v for k, v in orjson.loads(open(此处 / 'data\人频率6000000~7000000.json').read()).items() if v > 24}
人频率2 = {k: v for k, v in orjson.loads(open(此处 / 'data\人频率1~6400000.json').read()).items() if v > 64}
要测的人 = sorted(set(人频率1) | set(人频率2))
