import json
import hashlib
import itertools
from pathlib import Path

import orjson
from PIL import Image
from tqdm import tqdm

from 模型 import 模型池
from common import safe_name, 参数相同, 识别画师

sampler = 'DPM++ 2M'
scheduler = 'Karras'
steps = 20
cfg_scale = 7

存图文件夹 = Path('out_画风')
存图文件夹.mkdir(exist_ok=True)


def 标签源():
    while True:
        for a in ['1girl']:
            for b in ['white hair', 'black hair', 'blonde hair']:
                for c in ['twintails', 'long hair']:
                    for d in ['indoors', 'outdoors', None]:
                        yield [i for i in (a, b, c, d) if i]


def 评测模型(model, VAE, n_iter, *, use_tqdm=True, width=1024, height=1024, 图片缓存=False, model_type, 覆盖参数):
    from backend_diffusers import txt2img

    存档文件名 = f'savedata/画风_{model}_{width}_记录.json'
    if Path(存档文件名).exists():
        with open(存档文件名, 'r', encoding='utf8') as f:
            记录 = orjson.loads(f.read())
    else:
        记录 = []
    要测的画师 = json.load(open('data/画师top500.json', encoding='utf8'))
    iterator = range(n_iter)
    if use_tqdm:
        iterator = tqdm(iterator, ncols=80, desc=f'{width}-{model[:20]}')
    for index, tags, 画师 in zip(iterator, 标签源(), 要测的画师):
        if model_type == 'anima':
            prompt = ', '.join([i.replace('_', ' ') for i in (tags + ['@'+画师])])
        else:
            prompt = ', '.join([i.replace('_', ' ') for i in (tags + [画师])])
        参数 = {
            'prompt': prompt,
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
        for i in 记录:
            if 参数相同(i['参数'], 参数):
                break
        图s = txt2img({'n': 1} | 参数 | 覆盖参数, 缓存=图片缓存)
        md5 = hashlib.md5(prompt.encode()).hexdigest()
        for i, b in enumerate(图s):
            with open(存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png'), 'wb') as f:
                f.write(b)
        n = len(图s)
        图路径 = [存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png') for i in range(n)]
        预测 = [识别画师(Image.open(p)) for p in 图路径]
        录 = {
            '预测': 预测,
            '总数': n,
            '画师': 画师,
            '参数': 参数,
        }
        记录.append(录)
    with open(存档文件名, 'wb') as f:
        f.write(orjson.dumps(记录))


if __name__ == '__main__':
    白名单 = ['waiIllustriousSDXL_v160', 'noobaiXLNAIXL_epsilonPred11Version', 'novaAnimeXL_ilV160', 'chenkinNoobXLCKXL_v05', 'illustriousXL_v01', 'Illustrious-XL-v2.0', 'RUM-FLUX.2-klein-4B', 'Anima-Base-v1.0-Diffusers']
    for model in tqdm(模型池.values()):
        if model.类型 not in ('sdxl', 'rum-flux2klein', 'anima'):
            continue
        if model.名字 not in 白名单:
            continue
        评测模型(model.名字, model.vae, n_iter=450, 图片缓存=True, model_type=model.类型, 覆盖参数=model.额外参数 or {})
