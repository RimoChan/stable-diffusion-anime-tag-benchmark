import os
import re
import json
import base64
import itertools
from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm
from PIL import Image

import ml_danbooru


服务器地址 = f'http://127.0.0.1:7860'
要测的模型 = [
    ('anything-v4.5-pruned-fp32', 'anything-v4.0.vae.pt'),
    ('AOM3A1', 'orangemix.vae.pt'),
    ('AOM3A2', 'orangemix.vae.pt'),
    ('cosplaymix_v20', None),
    ('Counterfeit-V2.2', 'Counterfeit-V2.5.vae.pt'),
    ('novelailatest-pruned', 'novelailatest-pruned.vae.pt'),
    ('CounterfeitXL-V1.0', None),
    ('blue_pencil-XL-v0.3.1', None),
]
要测的标签 = ['twintails', 'red dress']

sampler = 'DPM++ 2M Karras'
seed = 1
steps = 30
width = 512
height = 512
cfg_scale = 7


def ml_danbooru标签(image_list: list[Union[str, bytes, os.PathLike]]) -> dict[str, dict[str, float]]:
    超d = {}
    for image in image_list:
        tags = ml_danbooru.get_tags_from_image(Image.open(image), threshold=0.5, keep_ratio=True)
        超d[image] = tags
    return 超d


def 上网(p, j=None, method='get'):
    r = getattr(requests, method)(p, json=j)
    r.reason = r.text[:4096]
    r.raise_for_status()
    return r.json()


def safe_name(s: str):
    return re.sub(r'[\\/:*?"<>|]', lambda m: str(ord(m.group()[0])), s)


存图文件夹 = Path('out')
存图文件夹.mkdir(exist_ok=True)

所有模型 = [i['model_name'] for i in 上网(f'{服务器地址}/sdapi/v1/sd-models')]
所有VAE = [i['model_name'] for i in 上网(f'{服务器地址}/sdapi/v1/sd-vae')]
assert set([i[0] for i in 要测的模型]) < set(所有模型)
assert set([i[1] for i in 要测的模型]) < set(所有VAE + [None])

if Path('记录.json').exists():
    with open('记录.json', 'r', encoding='utf8') as f:
        记录 = json.load(f)
else:
    记录 = []

for (model, VAE), 标签 in tqdm(itertools.product(要测的模型, 要测的标签), total=len(要测的模型) * len(要测的标签)):
    标签 = 标签.strip().replace(' ', '_')
    参数 = {
        'prompt': f'1 girl, {标签}',
        'negative_prompt': 'worst quality, low quality, blurry, greyscale, monochrome',
        'seed': seed,
        'width': width,
        'height': height,
        'steps': steps,
        'sampler_index': sampler,
        'cfg_scale': cfg_scale,
        'override_settings': {
            'sd_model_checkpoint': model,
            'sd_vae': VAE,
        },
    }
    skip = False
    for i in 记录:
        if i['标签'] == 标签 and i['参数'] == 参数:
            skip = True
            break
    if skip:
        continue
    数量参数 = {
        'batch_size': 4,
        'n_iter': 4,
    }
    r = 上网(f'{服务器地址}/sdapi/v1/txt2img', 数量参数 | 参数, 'post')
    图s = [base64.b64decode(b64) for b64 in r['images']]
    for i, b in enumerate(图s):
        with open(存图文件夹 / safe_name(f'{标签}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png'), 'wb') as f:
            f.write(b)
    n = len(图s)
    预测标签 = ml_danbooru标签([存图文件夹 / safe_name(f'{标签}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png') for i in range(n)])
    录 = {
        '分数': [i.get(标签, 0) for i in 预测标签.values()],
        '总数': n,
        '标签': 标签,
        '参数': 参数,
    }
    记录.append(录)
    with open('记录.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(记录, ensure_ascii=False, indent=4))
