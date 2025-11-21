import random
import base64
import itertools
from pathlib import Path

import orjson
from tqdm import tqdm

from 模型 import 模型池
from common import ml_danbooru标签, safe_name, 要测的标签, 参数相同
from backend_diffusers import txt2img

sampler = 'DPM++ 2M'
scheduler = 'Karras'
seed = 1
steps = 30
cfg_scale = 7

存图文件夹 = Path('out')
存图文件夹.mkdir(exist_ok=True)


def 评测模型(model, VAE, model_type, width, height, 覆盖参数) -> list[dict]:
    存档文件名 = f'savedata/单标签_{model}_{width}_记录.json'
    if Path(存档文件名).exists():
        with open(存档文件名, 'r', encoding='utf8') as f:
            记录 = orjson.loads(f.read())
    else:
        记录 = []
    for index, 标签 in enumerate(tqdm(要测的标签, ncols=70, desc=model[:10])):
        标签 = 标签.strip().replace(' ', '_')
        参数 = {
            'prompt': f'1 girl, {标签}',
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
            if i['标签'] == 标签:
                if 参数相同(i['参数'], 参数):
                    skip = True
                    break
                raise Exception(f'重复了！\n{i["参数"]}\n{参数}')
        if skip:
            continue
        数量参数 = {'n': 16}
        图s = txt2img(数量参数 | 参数 | 覆盖参数)
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
            '预测标签': {str(k): v for k, v in 预测标签.items()},
        }
        记录.append(录)
        if random.random() < 0.1 or index == len(要测的标签) - 1:
            with open(存档文件名, 'wb') as f:
                f.write(orjson.dumps(记录))
    return 记录


for model in tqdm(模型池.values()):
    if model.类型 == 'sd':
        size = 512
    else:
        size = 768
    评测模型(model.名字, model.vae, model.类型, width=size, height=size, 覆盖参数=model.额外参数 or {})
