import random
import base64
import hashlib
import itertools
from pathlib import Path

import orjson
from PIL import Image
from tqdm import tqdm

from common import 上网, ml_danbooru标签, safe_name, 服务器地址, check_model, 图像相似度, 要测的标签, 参数相同


要测的模型 = [
    ('anythingV3_fp16', 'anything-v4.0.vae.pt'),
    ('anything-v4.5-pruned-fp32', 'anything-v4.0.vae.pt'),
    ('AOM3A1', 'orangemix.vae.pt'),
    ('AOM3A2', 'orangemix.vae.pt'),
    ('aoaokoPVCStyleModel_pvcAOAOKO', 'novelailatest-pruned.vae.pt'),
    ('Aidv210AnimeIllustDiffusion_aidv28', 'vae-ft-mse-840000-ema-pruned.ckpt'),
    ('Aidv210AnimeIllustDiffusion_aidv210', 'vae-ft-mse-840000-ema-pruned.ckpt'),
    ('calicomix_v75', None),
    ('cosplaymix_v20', None),
    ('counterfeitV30_20', 'Counterfeit-V2.5.vae.pt'),
    ('Counterfeit-V2.2', 'Counterfeit-V2.5.vae.pt'),
    ('Counterfeit-V2.5_pruned', 'Counterfeit-V2.5.vae.pt'),
    ('novelailatest-pruned', 'novelailatest-pruned.vae.pt'),
    ('darkSushiMixMix_225D', 'vae-ft-mse-840000-ema-pruned.ckpt'),
    ('etherBluMix_etherBluMix5', 'kl-f8-anime2.ckpt'),
    ('bluePencil_v9', 'clearvae_v23.safetensors'),
    ('bluePencil_v10', 'clearvae_v23.safetensors'),
    ('Counterfeit-V3.0_fp16', 'kl-f8-anime2.ckpt'),
    ('AnythingV5Ink_ink', None),
    ('sweetfruit_melon.safetensors_v1.0', 'vae-ft-mse-840000-ema-pruned.ckpt'),
    ('cuteyukimixAdorable_midchapter', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_midchapter2', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_midchapter3', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_neochapter', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_neochapter2', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_neochapter3', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_specialchapter', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_naiV3style', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_kemiao', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_kemiaomiao', 'anything-v4.0.vae.pt'),
    ('cuteyukimixAdorable_echodimension', 'anything-v4.0.vae.pt'),
    ('cocotifacute_v20', 'novelailatest-pruned.vae.pt'),
    ('pastelMixStylizedAnime_pastelMixPrunedFP16', 'kl-f8-anime2.ckpt'),
    ('perfectWorld_v2Baked', None),
    ('perfectWorld_v6Baked', None),
    ('himawarimix_v100', None),
    ('meinamix_meinaV11', None),
    ('mixProV4_v4', 'novelailatest-pruned.vae.pt'),
    ('cetusMix_cetusVersion3', 'kl-f8-anime2.ckpt'),
    ('cetusMix_Coda2', 'kl-f8-anime2.ckpt'),
    ('cetusMix_Whalefall2', 'kl-f8-anime2.ckpt'),
    ('cetusMix_v4', 'kl-f8-anime2.ckpt'),
    ('cetusMix_cetusVersion2', 'kl-f8-anime2.ckpt'),
    ('sakuramochimix_v10', 'novelailatest-pruned.vae.pt'),
    ('sweetMix_v22Flat', 'blessed2.vae.safetensors'),
    ('ghostmix_v20Bakedvae', None),
    ('anyloraCheckpoint_novaeFp16', 'kl-f8-anime2.ckpt'),
    ('PVCStyleModelMovable_v20NoVae', 'vae-ft-mse-840000-ema-pruned.ckpt'),
    ('PVCStyleModelMovable_v30', 'vae-ft-mse-840000-ema-pruned.ckpt'),
    ('divineelegancemix_V9', 'MoistMix.vae.pt'),
    ('rainbowsweets_v20', None),
    ('rabbit_v7', 'novelailatest-pruned.vae.pt'),
    ('rimochan_random_mix', 'blessed2.vae.safetensors'),
    ('rimochan_random_mix_1.1', 'blessed2.vae.safetensors'),
    ('rimochan_random_mix_2.1', 'blessed2.vae.safetensors'),
    ('rimochan_random_mix_3.2', 'blessed2.vae.safetensors'),
    ('koji_v21', 'clearvae_v23.safetensors'),
    ('kaywaii_v50', 'clearvae_v23.safetensors'),
    ('kaywaii_v60', 'clearvae_v23.safetensors'),
    ('kaywaii_v70', 'clearvae_v23.safetensors'),
    ('kaywaii_v80', 'clearvae_v23.safetensors'),
    ('kaywaii_v85', 'clearvae_v23.safetensors'),
    ('kaywaii_v90', 'clearvae_v23.safetensors'),
    ('jitq_v20', 'blessed2.vae.safetensors'),
    ('jitq_v30', 'blessed2.vae.safetensors'),
    ('petitcutie_v15', 'blessed2.vae.safetensors'),
    ('petitcutie_v20', 'blessed2.vae.safetensors'),
    ('superInvincibleAnd_v2', 'blessed2.vae.safetensors'),
    ('ApricotEyes_v10', 'blessed2.vae.safetensors'),
    ('Yorunohitsuji-v1.0', 'novelailatest-pruned.vae.pt'),
    ('aiceKawaice_channel', 'blessed2.vae.safetensors'),
    ('coharumix_v6', 'blessed2.vae.safetensors'),
    ('irismix_v90', None),
    ('yetanotheranimemodel_v20', 'blessed2.vae.safetensors'),
    ('theWondermix_v12', 'blessed2.vae.safetensors'),
    ('animeIllustDiffusion_v052', 'sdxl_vae.safetensors'),
    ('animeIllustDiffusion_v061', 'sdxl_vae.safetensors'),
    ('CounterfeitXL-V1.0', None),
    ('counterfeitxl_v20', None),
    ('counterfeitxl_v25', None),
    ('hassakuXLSfwNsfwBeta_betaV01', None),
    ('reproductionSDXL_2v12', None),
    ('kohakuXLBeta_beta7', 'sdxl_vae.safetensors'),
    ('blue_pencil-XL-v0.3.1', None),
    ('PVCStyleModelFantasy_betaV10', 'sdxl_vae.safetensors'),
]

sampler = 'DPM++ 2M Karras'
steps = 30
width = 512
height = 512
cfg_scale = 7

存图文件夹 = Path('out_多标签')
存图文件夹.mkdir(exist_ok=True)

check_model(要测的模型)


if Path('savedata/记录_多标签.json').exists():
    with open('savedata/记录_多标签.json', 'r', encoding='utf8') as f:
        记录 = orjson.loads(f.read())
else:
    记录 = []


def 评测模型(model, VAE, m, n_iter, use_tqdm=True, savedata=True, extra_prompt='', seed=1, tags_seed=0, 计算相似度=True):
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
            'sampler_index': sampler,
            'cfg_scale': cfg_scale,
            'override_settings': {
                'sd_model_checkpoint': model,
                'sd_vae': VAE,
                'CLIP_stop_at_last_layers': 1,
            },
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
        r = 上网(f'{服务器地址}/sdapi/v1/txt2img', 数量参数 | 参数, 'post')
        图s = [base64.b64decode(b64) for b64 in r['images']]
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
        }
        if 计算相似度:
            相似度 = []
            for a, b in itertools.pairwise([Image.open(存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png')) for i in range(n)]):
                相似度.append(图像相似度(a, b))
            录['相似度'] = 相似度

        本地记录.append(录)
        记录.append(录)
    if savedata:
        with open('savedata/记录_多标签.json', 'wb') as f:
            f.write(orjson.dumps(记录))
    return 本地记录


if __name__ == '__main__':
    for (model, VAE), (m, n_iter) in tqdm([*itertools.product(要测的模型, ((2, 110), (4, 100), (8, 90), (16, 80), (32, 70), (64, 60), (128, 50)))]):
        评测模型(model, VAE, m, n_iter)
