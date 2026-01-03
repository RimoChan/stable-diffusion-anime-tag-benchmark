import random
import hashlib
import itertools
from pathlib import Path

import orjson
from PIL import Image
from tqdm import tqdm

from 模型 import 模型池
from common import ml_danbooru标签, safe_name, 图像相似度, 要测的标签, 参数相同

sampler = 'DPM++ 2M'
scheduler = 'Karras'
steps = 30
cfg_scale = 7

存图文件夹 = Path('out_多标签')
存图文件夹.mkdir(exist_ok=True)


def 评测模型(model, VAE, m, n_iter, *, use_tqdm=True, savedata=True, extra_prompt='', tags_seed=0, 计算相似度=True, width=512, height=512, 图片缓存=False, model_type, 覆盖参数):
    from backend_diffusers import txt2img

    存档文件名 = f'savedata/多标签_{model}_{width}_记录_v2.json'
    if Path(存档文件名).exists():
        with open(存档文件名, 'r', encoding='utf8') as f:
            记录 = orjson.loads(f.read())
    else:
        记录 = []
    rd = random.Random(tags_seed)
    本地记录 = []
    iterator = range(n_iter)
    if use_tqdm:
        iterator = tqdm(iterator, ncols=80, desc=f'{m}-{width}-{model[:20]}')
    for index in iterator:
        标签组 = rd.sample(要测的标签, m)
        标签组 = [i.strip() for i in 标签组]
        参数 = {
            'prompt': f'1 girl, {", ".join(标签组)}'+extra_prompt,
            'negative_prompt': 'worst quality, low quality',
            'seed': index,
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
            if 参数相同(i['参数'], 参数):
                skip = True
                break
        if skip:
            本地记录.append(i)
            continue
        数量参数 = {'n': 4}
        图s = txt2img(数量参数 | 参数 | 覆盖参数, 缓存=图片缓存)
        md5 = hashlib.md5(str(标签组).encode()).hexdigest()
        for i, b in enumerate(图s):
            with open(存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png'), 'wb') as f:
                f.write(b)
        n = len(图s)
        预测标签 = ml_danbooru标签([存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png') for i in range(n)])

        标签组 = [i.replace(' ', '_') for i in 标签组]
        录 = {
            '分数': [[i.get(j, 0) for j in 标签组] for i in 预测标签.values()],
            '总数': n,
            '标签组': 标签组,
            '参数': 参数,
            '预测标签': {str(k): v for k, v in 预测标签.items()},
        }
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
    for model in tqdm(模型池.values()):
        if model.类型 in ('sdxl', 'flux.1s', 'flux.1d', 'sd3', 'neta-lumina', 'sana'):
            测试组合 = [(8, 100, 1024), (32, 100, 768), (32, 100, 1024), (32, 25, 1280)]
        elif model.类型 in ('sd', ):
            测试组合 = [(8, 100, 512), (32, 100, 512), (32, 100, 768)]
        else:
            continue
        for m, n_iter, size in 测试组合:
            评测模型(model.名字, model.vae, m, n_iter, width=size, height=size, 图片缓存=True, model_type=model.类型, 覆盖参数=model.额外参数 or {})
