import io
from pathlib import Path

import torch
from transformers import T5EncoderModel
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline, FluxTransformer2DModel, FluxPipeline, DPMSolverMultistepScheduler, StableDiffusionXLPipeline, EulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline, Lumina2Pipeline, SanaPipeline, ZImagePipeline, Flux2KleinPipeline
from compel import Compel, ReturnedEmbeddingsType
from safetensors import safe_open
from optimum.quanto import freeze, qfloat8, quantize

import rimo_storage.cache

from DeepCache import DeepCacheSDHelper

torch._dynamo.config.cache_size_limit = 64

model_dirs = [
    'S:/Stable-diffusion-models',
    'R:/stable-diffusion-webui-master/models',
    'R:/models',
    'X:/stable-diffusion-webui-master/models',
    'X:/models',
    'C:/Users/Administrator/Desktop/WAI-illustrious-1.5',
    'Z:/models/Diffusion',
    'Y:/models/Diffusion',
]


_l = torch.load
def _假load(*t, **d):
    d.pop('weights_only', None)
    return _l(*t, **d)
torch.load = _假load


class 超StableDiffusionKDiffusionPipeline:
    def __init__(self, path, vae_path=None, 串行化vae=True):
        p = {}
        if vae_path:
            p['vae'] = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16, weights_only=False).to("cuda")
        pipe0 = StableDiffusionPipeline.from_single_file(path, **p, torch_dtype=torch.float16)
        c = pipe0.components
        c.pop('image_encoder')
        self._pipe = StableDiffusionKDiffusionPipeline(**c).to('cuda')
        self._pipe.set_scheduler('sample_dpmpp_2m')
        if 串行化vae:
            self._pipe.enable_vae_slicing()
        self._compel = Compel(tokenizer=self._pipe.tokenizer, text_encoder=self._pipe.text_encoder, truncate_long_prompts=False)

    def __call__(
        self,
        prompt: list[str],
        negative_prompt: list[str],
        **d,
    ):
        conditioning = self._compel(prompt)
        negative_conditioning = self._compel(negative_prompt)
        [conditioning, negative_conditioning] = self._compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
        return self._pipe.__call__(
            prompt_embeds=conditioning,
            negative_prompt_embeds=negative_conditioning,
            **d,
        )


class 超StableDiffusionXLPipeline:
    def __init__(self, path, vae_path=None, 串行化vae=True, 使用deepcache=False, 保存中间结果=False):
        p = {}
        if vae_path:
            p['vae'] = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16, scaling_factor=0.13025).to("cuda")
        self._pipe = StableDiffusionXLPipeline.from_single_file(
            path,
            torch_dtype=torch.float16,
            **p,
        ).to("cuda")
        _keys = safe_open(path, framework='torch').keys()
        if 'v_pred' in _keys:
            self._pipe.scheduler = EulerDiscreteScheduler.from_config(self._pipe.scheduler.config, prediction_type='v_prediction', rescale_betas_zero_snr=True)
        else:
            self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(self._pipe.scheduler.config)
        self._pipe.set_progress_bar_config(disable=True)
        if 串行化vae:
            self._pipe.enable_vae_slicing()
        if 使用deepcache:
            helper = DeepCacheSDHelper(pipe=self._pipe)
            helper.set_params(cache_interval=2)
            helper.enable()
        if 保存中间结果:
            _s = self._pipe.scheduler.step
            def 假step(noise_pred, t, latents, **d):
                m = self._pipe.scheduler.timesteps.max()
                denoised_latents = (latents-noise_pred * (t / m))[0]    # 这个实现有点问题，不过还算能看，就先这样吧
                denoised_latents = denoised_latents.reshape([1, *denoised_latents.shape])
                image = self._pipe.image_processor.postprocess(
                    self._pipe.vae.to(torch.float32).decode(denoised_latents.to(torch.float32) / self._pipe.vae.config.scaling_factor, return_dict=False)[0],
                    output_type='pil',
                )[0]
                self.step_image.append(image)
                self._pipe.vae.to(torch.float16)
                return _s(noise_pred, t, latents, **d)
            self._pipe.scheduler.step = 假step
        self.compel = Compel(truncate_long_prompts=False, tokenizer=[self._pipe.tokenizer, self._pipe.tokenizer_2], text_encoder=[self._pipe.text_encoder, self._pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

    def __call__(
        self,
        prompt: list[str],
        negative_prompt: list[str],
        **d,
    ):
        conditioning, pooled = self.compel(prompt)
        negative_embed, negative_pooled = self.compel(negative_prompt)
        [conditioning, negative_embed] = self.compel.pad_conditioning_tensors_to_same_length([conditioning, negative_embed])
        return self._pipe(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=negative_embed,
            negative_pooled_prompt_embeds=negative_pooled,
            **d,
        )


def find_file(s) -> Path:
    if not s:
        return None
    s = s.removesuffix('.safetensors')
    a = []
    for base in model_dirs:
        候选 = [*Path(base).glob('**/*.safetensors')] + [*Path(base).glob('**/*.pt')] + [*Path(base).glob('**/*.ckpt')] + [*Path(base).glob('**/*/')]
        for i in 候选:
            if i.stem == s or i.name == s:
                a.append(i)
    if not a:
        raise Exception(f'找不到{s}！')
    return a[0]


def pipeline0(model_type, path, vae_path) -> 超StableDiffusionKDiffusionPipeline | 超StableDiffusionXLPipeline | FluxPipeline:
    if model_type == 'sd':
        return 超StableDiffusionKDiffusionPipeline(path, vae_path)
    elif model_type == 'sdxl':
        return 超StableDiffusionXLPipeline(path, vae_path)
    elif model_type in ('flux.1s', 'flux.1d'):
        if model_type == 'flux.1s':
            repo = "black-forest-labs/FLUX.1-schnell"
        elif model_type == 'flux.1d':
            repo = "black-forest-labs/FLUX.1-dev"
        transformer = FluxTransformer2DModel.from_single_file(path, torch_dtype=torch.bfloat16).to('cuda')
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
        text_encoder_2 = T5EncoderModel.from_pretrained(repo, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to('cuda')
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
        pipe = FluxPipeline.from_pretrained(repo, transformer=None, text_encoder_2=None, torch_dtype=torch.bfloat16)
        pipe.set_progress_bar_config(disable=True)
        pipe.transformer = transformer
        pipe.text_encoder_2 = text_encoder_2
        pipe.vae.enable_slicing()
        pipe = pipe.to('cuda')
        return pipe
    elif model_type == 'sd3':
        model = SD3Transformer2DModel.from_single_file(path, torch_dtype=torch.bfloat16).to('cuda')
        pipe = StableDiffusion3Pipeline.from_pretrained('stabilityai/stable-diffusion-3.5-medium', transformer=model, torch_dtype=torch.bfloat16).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        return pipe
    elif model_type == 'neta-lumina':
        pipe = Lumina2Pipeline.from_pretrained(path, torch_dtype=torch.bfloat16).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        return pipe
    elif model_type == 'sana':
        pipe = SanaPipeline.from_pretrained(path, variant="fp16", torch_dtype=torch.float16)
        pipe.to("cuda")
        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)
        pipe.set_progress_bar_config(disable=True)
        return pipe
    elif model_type == 'z-image':
        pipe = ZImagePipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        return pipe
    elif model_type == 'flux2klein':
        pipe = Flux2KleinPipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
        pipe.set_progress_bar_config(disable=True)
        pipe.to("cuda")
        return pipe
    else:
        raise Exception(f'不认识模型类型{model_type}！')


def pipeline(model_type, path, vae_path, lora_path) -> 超StableDiffusionKDiffusionPipeline | 超StableDiffusionXLPipeline | FluxPipeline:
    p = pipeline0(model_type, path, vae_path)
    if lora_path:
        print(lora_path)
        p._pipe.load_lora_weights(lora_path, adapter_name="Q")
        p._pipe.set_adapters(["Q"], adapter_weights=[1.0])
        p._pipe.fuse_lora(adapter_names=["Q"], lora_scale=1.0)
        p._pipe.unload_lora_weights()
    return p


_loaded_pipeline = None
def get_pipeline(model_type, model, vae, lora):
    global _loaded_pipeline
    if _loaded_pipeline and _loaded_pipeline[1] == (model_type, model, vae, lora):
        return _loaded_pipeline[0]
    torch.cuda.empty_cache()
    _loaded_pipeline = pipeline(model_type, find_file(model), find_file(vae), find_file(lora)), (model_type, model, vae, lora)
    torch.cuda.empty_cache()
    return _loaded_pipeline[0]


def txt2img(p: dict, 缓存=True) -> list[bytes]:
    if 缓存:
        return _txt2img_缓存(p)
    else:
        return _txt2img(p)


@rimo_storage.cache.disk_cache(serialize='pickle')
def _txt2img_缓存(p: dict) -> list[bytes]:
    return _txt2img(p)


def _txt2img(p: dict) -> list[bytes]:
    参数 = {}

    model_type = p.pop('model_type')

    参数['prompt'] = p.pop('prompt')
    参数['negative_prompt'] = p.pop('negative_prompt')
    参数['width'] = p.pop('width')
    参数['height'] = p.pop('height')
    参数['num_inference_steps'] = p.pop('steps')
    参数['guidance_scale'] = p.pop('cfg_scale')

    assert p.pop('sampler_name') == 'DPM++ 2M'
    assert p.pop('scheduler') == 'Karras'
    参数['use_karras_sigmas'] = True

    override_settings = p.pop('override_settings', None)

    assert 'batch_size' not in p and 'n_iter' not in p, 'batch_size和n_iter废弃了，传n'
    n = p.pop('n')
    if model_type in ('sdxl', 'sd'):
        batch_size = 4
    else:
        batch_size = 1
    assert n % batch_size == 0
    n_iter = n // batch_size

    if model_type == 'neta-lumina':
        参数['prompt'] = 'You are an assistant designed to generate anime images based on textual prompts. <Prompt Start> ' + 参数['prompt']
        参数['negative_prompt'] = 'You are an assistant designed to generate anime images based on textual prompts. <Prompt Start> ' + 参数['negative_prompt']

    参数['prompt'] = [参数['prompt']] * batch_size
    参数['negative_prompt'] = [参数['negative_prompt']] * batch_size
    seed = p.pop('seed')

    for k in ['cfg_trunc_ratio', 'cfg_normalization']:
        if k in p:
            参数[k] = p.pop(k)

    assert not p, f'剩下参数{p}不知道怎么转换……'

    pipe = get_pipeline(model_type, override_settings['sd_model_checkpoint'], override_settings['sd_vae'], override_settings.get('lora'))

    if model_type in ('flux.1s', 'flux.1d', 'sd3', 'sana', 'z-image', 'flux2klein'):
        参数.pop('negative_prompt')
        参数.pop('use_karras_sigmas')
    if model_type == 'neta-lumina':
        参数.pop('use_karras_sigmas')
    res = []
    for i in range(n_iter):
        res.extend(pipe(
            **参数,
            generator=[torch.Generator(device='cuda').manual_seed(j) for j in range(seed + batch_size*i, seed + batch_size*(i+1))],
        ).images)
        torch.cuda.empty_cache()
    res_b = []
    for image in res:
        b = io.BytesIO()
        image.save(b, 'png')
        res_b.append(b.getvalue())
    return res_b
