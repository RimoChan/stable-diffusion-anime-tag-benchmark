import random
import base64
import hashlib
import itertools
from pathlib import Path

import orjson
from PIL import Image
from tqdm import tqdm

from common import ml_danbooru标签, safe_name, 图像相似度, 要测的标签, 参数相同, 图像质量, 模型数据
from backend_diffusers import txt2img

sampler = 'DPM++ 2M'
scheduler = 'Karras'
steps = 30
cfg_scale = 7

存图文件夹 = Path('out_多标签')
存图文件夹.mkdir(exist_ok=True)


def 评测模型(model, VAE, m, n_iter, *, use_tqdm=True, savedata=True, extra_prompt='', seed=1, tags_seed=0, 计算相似度=True, width=512, height=512, 图片缓存=False,  计算图像质量=True, model_type):
    存档文件名 = f'savedata/多标签_{model}_{width}_记录.json'
    if Path(存档文件名).exists():
        with open(存档文件名, 'r', encoding='utf8') as f:
            记录 = orjson.loads(f.read())
    else:
        记录 = []
    rd = random.Random(tags_seed)
    本地记录 = []
    iterator = range(n_iter)
    if use_tqdm:
        iterator = tqdm(iterator, ncols=70, desc=f'{m}-{model[:10]}')
    for index in iterator:
        标签组 = rd.sample(要测的标签, m)
        标签组 = [i.strip().replace(' ', '_') for i in 标签组]
        参数 = {
            'prompt': f'1 girl, {", ".join(标签组)}'+extra_prompt,
            'negative_prompt': 'worst quality, low quality, blurry, greyscale, monochrome',
            'seed': seed,
            'width': width,
            'height': height,
            'steps': steps,
            'sampler_name': sampler,
            'scheduler': scheduler,
            'cfg_scale': cfg_scale,
            'override_settings': {
                'sd_model_checkpoint': model,
                'sd_vae': VAE,
                'CLIP_stop_at_last_layers': 1,
            },
            'model_type': model_type,
        }
        skip = False
        for i in 记录:
            if i['标签组'] == 标签组 and 参数相同(i['参数'], 参数):
                skip = True
                break
        if skip:
            本地记录.append(i)
            continue
        数量参数 = {
            'batch_size': 4,
            'n_iter': 2,
        }
        图s = txt2img(数量参数 | 参数, 缓存=图片缓存)
        md5 = hashlib.md5(str(标签组).encode()).hexdigest()
        for i, b in enumerate(图s):
            with open(存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png'), 'wb') as f:
                f.write(b)
        n = len(图s)
        预测标签 = ml_danbooru标签([存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png') for i in range(n)])

        录 = {
            '分数': [[i.get(j, 0) for j in 标签组] for i in 预测标签.values()],
            '总数': n,
            '标签组': 标签组,
            '参数': 参数,
            '预测标签': {str(k): v for k, v in 预测标签.items()},
        }
        if 计算图像质量:
            质量 = 图像质量([Image.open(存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png')) for i in range(n)])
            录['质量'] = 质量
        if 计算相似度:
            相似度 = []
            for a, b in itertools.pairwise([Image.open(存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png')) for i in range(n)]):
                相似度.append(图像相似度(a, b))
            录['相似度'] = 相似度

        本地记录.append(录)
        记录.append(录)
    if savedata:
        with open(存档文件名, 'wb') as f:
            f.write(orjson.dumps(记录))
    return 本地记录


if __name__ == '__main__':
    for (model, VAE, 简称, model_type, 弃用), (m, n_iter, size) in tqdm([*itertools.product(模型数据, ((2, 110, 768), (8, 90, 768), (32, 70, 768), (32, 70, 1024)))]):
        if 弃用:
            continue
        评测模型(model, VAE, m, n_iter, width=size, height=size, 图片缓存=True, model_type=model_type)
