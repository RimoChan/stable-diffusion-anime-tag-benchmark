import json
import random
import base64
import hashlib
import itertools
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from common import 上网, ml_danbooru标签, safe_name, 服务器地址, check_model, 图像相似度, 要测的标签


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
    ('darkSushiMixMix_225D', 'vae-ft-mse-840000-ema-pruned.ckpt'),
    ('etherBluMix_etherBluMix5', 'kl-f8-anime2.ckpt'),
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
    ('cuteyukimixAdorable_kemiao', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_kemiaomiao', 'anything-v4.0.vae.pt'),
    ('cocotifacute_v20', 'novelailatest-pruned.vae.pt'),
    ('perfectWorld_v2Baked', None),
    ('perfectWorld_v6Baked', None),
    ('meinamix_meinaV11', None),
    ('mixProV4_v4', 'novelailatest-pruned.vae.pt'),
    ('cetusMix_cetusVersion3', 'kl-f8-anime2.ckpt'),
    ('cetusMix_Coda2', 'kl-f8-anime2.ckpt'),
    ('cetusMix_Whalefall2', 'kl-f8-anime2.ckpt'),
    ('cetusMix_v4', 'kl-f8-anime2.ckpt'),
    ('sakuramochimix_v10', 'novelailatest-pruned.vae.pt'),
    ('ghostmix_v20Bakedvae', None),
    ('anyloraCheckpoint_novaeFp16', 'kl-f8-anime2.ckpt'),
    ('PVCStyleModelMovable_v20NoVae', 'vae-ft-mse-840000-ema-pruned.ckpt'),
    ('divineelegancemix_V9', 'MoistMix.vae.pt'),
    ('rainbowsweets_v20', None),
    ('rabbit_v7', 'novelailatest-pruned.vae.pt'),
    ('koji_v21', 'clearvae_v23.safetensors'),
    ('kaywaii_v50', 'clearvae_v23.safetensors'),
    ('kaywaii_v60', 'clearvae_v23.safetensors'),
    ('kaywaii_v70', 'clearvae_v23.safetensors'),
    ('kaywaii_v80', 'clearvae_v23.safetensors'),
    ('CounterfeitXL-V1.0', None),
    ('counterfeitxl_v20', None),
    ('counterfeitxl_v25', None),
]

sampler = 'DPM++ 2M Karras'
seed = 1
steps = 30
width = 512
height = 512
cfg_scale = 7

存图文件夹 = Path('out_多标签')
存图文件夹.mkdir(exist_ok=True)

check_model(要测的模型)


if Path('savedata/记录_多标签.json').exists():
    with open('savedata/记录_多标签.json', 'r', encoding='utf8') as f:
        记录 = json.load(f)
else:
    记录 = []


for (model, VAE), m, i in tqdm([*itertools.product(要测的模型, (2, 4, 8, 16, 32, 64, 128), range(50))]):
    if i == 0:
        random.seed(0)

    标签组 = random.sample(要测的标签, m)
    标签组 = [i.strip().replace(' ', '_') for i in 标签组]

    参数 = {
        'prompt': f'1 girl, {", ".join(标签组)}',
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
        if i['标签组'] == 标签组 and i['参数'] == 参数:
            skip = True
            break
    if skip:
        continue
    数量参数 = {
        'batch_size': 4,
        'n_iter': 2,
    }
    r = 上网(f'{服务器地址}/sdapi/v1/txt2img', 数量参数 | 参数, 'post')
    图s = [base64.b64decode(b64) for b64 in r['images']]
    md5 = hashlib.md5(str(标签组).encode()).hexdigest()
    for i, b in enumerate(图s):
        with open(存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png'), 'wb') as f:
            f.write(b)
    n = len(图s)
    预测标签 = ml_danbooru标签([存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png') for i in range(n)])

    相似度 = []
    for a, b in itertools.pairwise([Image.open(存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png')) for i in range(n)]):
        相似度.append(图像相似度(a, b))

    录 = {
        '分数': [[i.get(j, 0) for j in 标签组] for i in 预测标签.values()],
        '相似度': 相似度,
        '总数': n,
        '标签组': 标签组,
        '参数': 参数,
    }
    记录.append(录)
    with open('savedata/记录_多标签.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(记录, ensure_ascii=False, indent=2))
