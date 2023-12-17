import random
import base64
import itertools
from pathlib import Path

import orjson
from tqdm import tqdm

from common import 缓存上网, ml_danbooru标签, safe_name, 服务器地址, check_model, 要测的标签


要测的模型 = [
    ('anythingV3_fp16', 'anything-v4.0.vae.pt'),
    ('anything-v4.5-pruned-fp32', 'anything-v4.0.vae.pt'),
    ('AOM3A1', 'orangemix.vae.pt'),
    ('AOM3A2', 'orangemix.vae.pt'),
    ('aoaokoPVCStyleModel_pvcAOAOKO', 'novelailatest-pruned.vae.pt'),
    ('cosplaymix_v20', None),
    ('counterfeitV30_20', 'Counterfeit-V2.5.vae.pt'),
    ('Counterfeit-V2.2', 'Counterfeit-V2.5.vae.pt'),
    ('Counterfeit-V2.5_pruned', 'Counterfeit-V2.5.vae.pt'),
    ('novelailatest-pruned', 'novelailatest-pruned.vae.pt'),
    ('CounterfeitXL-V1.0', None),
    ('blue_pencil-XL-v0.3.1', None),
    ('bluePencil_v9', 'clearvae_v23.safetensors'),
    ('bluePencil_v10', 'clearvae_v23.safetensors'),
    ('Counterfeit-V3.0_fp16', 'kl-f8-anime2.ckpt'),
    ('AnythingV5Ink_ink', None),
    ('sweetfruit_melon.safetensors_v1.0', 'vae-ft-mse-840000-ema-pruned.ckpt'),
    ('cuteyukimixAdorable_midchapter2', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_midchapter3', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_neochapter3', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_specialchapter', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_naiV3style', 'anything-v4.0.vae.pt'),
    ('cocotifacute_v20', 'novelailatest-pruned.vae.pt'),
    ('perfectWorld_v2Baked', None),
    ('perfectWorld_v6Baked', None),
    ('meinamix_meinaV11', None),
    ('cetusMix_Whalefall2', 'kl-f8-anime2.ckpt'),
    ('cetusMix_v4', 'kl-f8-anime2.ckpt'),
    ('sakuramochimix_v10', 'novelailatest-pruned.vae.pt'),
    ('ghostmix_v20Bakedvae', None),
    ('anyloraCheckpoint_novaeFp16', 'kl-f8-anime2.ckpt'),
    ('PVCStyleModelMovable_v20NoVae', 'vae-ft-mse-840000-ema-pruned.ckpt'),
    ('divineelegancemix_V9', 'MoistMix.vae.pt'),
    ('kaywaii_v50', 'clearvae_v23.safetensors'),
    ('kaywaii_v80', 'clearvae_v23.safetensors'),
]


sampler = 'DPM++ 2M Karras'
seed = 1
steps = 30
width = 512
height = 512
cfg_scale = 7

存图文件夹 = Path('out')
存图文件夹.mkdir(exist_ok=True)

check_model(要测的模型)


if Path('savedata/记录.json').exists():
    with open('savedata/记录.json', 'r', encoding='utf8') as f:
        记录 = orjson.loads(f.read())
else:
    记录 = []

for index, ((model, VAE), 标签) in enumerate(tqdm(itertools.product(要测的模型, 要测的标签), total=len(要测的模型) * len(要测的标签), ncols=70)):
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
    r = 缓存上网(f'{服务器地址}/sdapi/v1/txt2img', 数量参数 | 参数, 'post')
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
        '预测标签': {str(k): v for k, v in 预测标签.items()},
    }
    记录.append(录)
    if random.random() < 0.05 or index == len(要测的模型) * len(要测的标签) - 1:
        with open('savedata/记录.json', 'wb') as f:
            f.write(orjson.dumps(记录))
