import os
import re
import base64
from pathlib import Path
from typing import Union, Optional, Iterable

import cv2
import orjson
import requests
import numpy as np
from PIL import Image
import rimo_storage.cache
from transformers import AutoProcessor, AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, pipeline


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


def txt2img(p: dict, 缓存=True) -> list[bytes]:
    if 缓存:
        r = 缓存上网(f'{服务器地址}/sdapi/v1/txt2img', p, 'post')
    else:
        r = 上网(f'{服务器地址}/sdapi/v1/txt2img', p, 'post')
    return [base64.b64decode(b64) for b64 in r['images']]


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
    a = a.copy()
    b = b.copy()
    a['override_settings'].pop('CLIP_stop_at_last_layers', None)
    a.pop('model_type', None)
    b['override_settings'].pop('CLIP_stop_at_last_layers', None)
    b.pop('model_type', None)
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


要测的标签 = ['1girl', 'solo', 'highres', 'long hair', 'breasts', 'looking at viewer', 'blush', 'smile', 'short hair', 'open mouth', 'bangs', 'blue eyes', 'skirt', 'blonde hair', 'large breasts', 'brown hair', 'simple background', 'black hair', 'hair ornament', 'absurdres', 'red eyes', 'thighhighs', 'gloves', '1boy', 'long sleeves', 'white background', 'dress', 'original', 'ribbon', 'bow', 'navel', '2girls', 'holding', 'animal ears', 'cleavage', 'hair between eyes', 'bare shoulders', 'twintails', 'brown eyes', 'medium breasts', 'jewelry', 'sitting', 'very long hair', 'underwear', 'closed mouth', 'nipples', 'school uniform', 'green eyes', 'blue hair', 'standing', 'purple eyes', 'collarbone', 'panties', 'tail', 'monochrome', 'swimsuit', 'full body', 'closed eyes', 'hair ribbon', 'yellow eyes', 'ponytail', 'upper body', 'purple hair', 'pink hair', 'ass', 'braid', 'comic', 'flower', 'ahoge', ':d', 'hair bow', 'greyscale', 'white hair', 'pantyhose', 'bikini', 'sidelocks', 'thighs', 'nude', 'red hair', 'cowboy shot', 'grey hair', 'pleated skirt', 'multicolored hair', 'hairband', 'earrings', 'small breasts', 'boots', 'outdoors', 'lying', 'censored', 'frills', 'parted lips', 'detached sleeves', 'one eye closed', 'japanese clothes', 'green hair', 'wings', 'open clothes', 'necktie', 'sky', 'horns', 'penis', 'glasses', 'shorts', 'barefoot', 'teeth', 'pussy', 'serafuku', 'solo focus', 'day', 'alternate costume', 'choker', 'tongue', 'pointy ears', 'socks', 'elbow gloves', 'hairclip', 'fang', 'midriff', 'striped', 'belt', 'looking back', 'collared shirt', 'sword', 'official art', 'black thighhighs', 'cloud', 'indoors', 'cat ears', 'tears', 'hair flower', 'signature', 'spread legs', 'cum', '2boys', 'hood', 'sex', 'miniskirt', 'tongue out', 'on back', 'fingerless gloves', 'blunt bangs', 'bowtie', 'armpits', 'pink eyes', 'medium hair', 'sailor collar', 'kimono', 'silver hair', 'water', 'necklace', 'black legwear', 'off shoulder', 'chibi', 'hair bun', 'clothes lift', 'from behind', 'cape', 'scarf', 'bra', 'yuri', 'white dress', 'rabbit ears', 'white panties', 'mole', 'hair over one eye', 'grin', 'uniform', 'huge breasts', ':o', 'blurry', 'black eyes', 'apron', 'looking at another', 'vest', 'black dress', 'mosaic censoring', 'arm up', 'high heels', 'twin braids', 'shiny hair', 'arms up', 'flat chest', 'side ponytail', 'bracelet', 'collar', 'covered nipples', 'from side', 'aqua eyes', 'white thighhighs', 'leotard', 'two side up', 'sketch', 'lips', 'zettai ryouiki', 'coat', 'streaked hair', 'v-shaped eyebrows', 'neckerchief', 'crop top', 'head tilt', 'see-through', 'orange eyes', 'white legwear', 'gradient', 'hand on hip', 'gun', 'wrist cuffs', 'sleeves past wrists', 'looking to the side', 'torn clothes', 'maid', 'grey eyes', 'parted bangs', 'kneehighs', 'maid headdress', 'one-piece swimsuit', 'cosplay', 'petals', 'pubic hair', 'symbol-shaped pupils', 'fox ears', 'gradient background', 'loli', 'short shorts', 'ascot', 'dutch angle', 'eyelashes', 'bar censor', 'no humans', 'bare legs', 'mole under eye', 'dress shirt', 'sparkle', 'kneeling', 'lowres', 'single braid', 'bodysuit', 'v', 'no bra', 'saliva', 'bell', 'double bun', 'uncensored', 'military uniform', 'blood', 'hoodie', 'sideboob', 'scan', 'tattoo', '4koma', 'pussy juice', 'profile', 'gradient hair', 'makeup', 'neck ribbon', 'leaning forward', 'mask', 'multiple views', 'no panties', 'witch hat', 'capelet', 'anus', ':3', 'alternate hairstyle', 'detached collar', 'underboob', 'night', 'floating hair', 'depth of field', 'buttons', 'sleeveless dress', '^ ^', 'toes', 'cameltoe', 'blue dress', 'fox tail', 'feet out of frame', 'bottomless', 'nose blush', 'swept bangs', 'rose', 'fake animal ears', 'holding hands', 'facial hair', 'turtleneck', 'siblings', 'headphones', 'ocean', 'heterochromia', 'arm support', 'low twintails', 'halterneck', 'umbrella', 'frown', 'beret', 'pov', 'thigh boots', 'embarrassed', 'one side up', 'fangs', 'back', 'from above', 'watermark', 'garter straps', 'ass visible through thighs', 'wavy hair', 'upper teeth', 'on side', 'plaid skirt', 'transparent background', 'wariza', 'traditional media', 'mouth hold', 'chair', 'beach', 'parody', 'bandages', 'looking away', 'blush stickers', 'expressionless', 'drill hair', 'chinese clothes', 'grabbing', 'thick thighs', 'arms behind back', 'obi', 'halo', 'heart-shaped pupils', 'clothes pull', 'topless', 'pantyshot', 'thigh gap', 'looking down', 'short dress', 'skirt lift', 'eyepatch', 'magical girl', 'floral print', 'crossed arms', 'piercing', 'wavy mouth', 'hair intakes', 'border', 'formal', 'moon', 'leg up', 'half-closed eyes', 'from below', 'cleavage cutout', 'cover', 'happy', 'oral', 'red dress', 'squatting', 'sunglasses', 'testicles', 'school swimsuit', 'bdsm', 'trembling', 'blazer', 'wolf ears', 'standing on one leg', 'game cg', 'light brown hair', 'backpack', 'bob cut', 'eyes visible through hair', 'knee boots', 'lingerie', 'breast grab', 'demon girl', 'frilled dress', 'cardigan', 'bat wings', 'crossed legs', ';d', 'hood down', 'antenna hair', 'outstretched arms', 'crying', 'tank top', 'own hands together', 'polka dot', 'suit', 'aged down', 'undressing', 'crown', 'tiara', 'bent over', 'breasts out', 'high ponytail', 'light smile', 'horse ears', 'animated', 'looking up', 'straddling', 'black hairband', 'x hair ornament', '> <', 'wing collar', 'plant', 'on stomach', 'hair bobbles', 'outstretched arm', 'areolae', 'lipstick', 'short twintails', 'bondage', 'letterboxed', 'girl on top', 'lifted by self', 'cat girl', 'pointing', 'revision', 'slit pupils', 'all fours', 'spiked hair', '^^^', 'sisters', 'hand on own chest', 'panty pull', 'cherry blossoms', 'blouse', 'clenched teeth', 'goggles', 'brooch', 'cover page', 'bike shorts', 'casual', 'loafers', 'demon horns', 'elf', 't-shirt', 'fox girl', 'pink panties', 'building', 'motion lines', 'horse girl', 'wristband', 'messy hair', 'striped panties', 'between breasts', 'breast press', 'surprised', 'group sex', 'child', 'drooling', 'clenched hand', 'fishnets', 'hakama', 'rope', 'candy', 'genderswap', 'facial', 'side-tie panties', 'covering', 'foreshortening', 'nature', 'dog ears', 'adapted costume', 'portrait', 'anal', 'pink dress', 'demon tail', 'peaked cap', 'armband', 'waist apron', 'convenient censoring', 'night sky', 'ejaculation', 'china dress', 'arms behind head', 'alice margatroid', 'tokin hat', 'bandaid', 'microphone', 'bara', 'lace', 'mole under mouth', 'front-tie top', 'bow panties', 'strapless dress', 'hakama skirt', 'yaoi', 'glowing eyes', 'anger vein', 'no headwear', 'straight hair', 'breasts apart', 'christmas', 'hair flaps', 'hair over shoulder', 'cum on breasts', 'twin drills', 'hooded jacket', 'torn legwear', ':<', 'angry', 'facing viewer', 'cloak', 'eyewear on head', 'side braid', 'corset', 'micro bikini', 'bright pupils', 'mary janes', 'index finger raised', 'purple dress', ':p', 'tareme', 'paizuri', 'close-up', 'french braid', 'striped thighhighs', 'puffy nipples', 'furry', 'neck bell', 'full moon', 'tsurime', 'forehead', 'eye contact', 'areola slip', 'upskirt', 'striped legwear', 'seiza', 'genderswap (mtf)', 'lens flare', 'armlet', 'hand on own face', 'between legs', 'side slit', 'horse tail', 'handgun', 'pendant', 'camisole', 'skin fang', 'masturbation', 'alcohol', 'handjob', 'low ponytail', 'heavy breathing', 'cross-laced footwear', 'hair scrunchie', 'cowgirl position', 'santa hat', 'highleg leotard', 'breast hold', 'personification', 'clenched hands', 'forest', 'dress lift', 'halloween', 'cropped legs', 'wide hips', 'high heel boots', 'doujin cover', 'low-tied long hair', 'hood up', 'legs up', 'hair rings', 'backlighting', 'pencil skirt', 'jpeg artifacts', 'fish', 'clothed sex', 'sleeves past fingers', 'starry sky', 'half updo', 'spot color', 'panties under pantyhose', 'tearing up', 'asymmetrical legwear', 'kemonomimi mode', 'sailor dress', 'akemi homura', 'rain', 'garter belt', 'dual wielding', 'walking', 'cuffs', 'fingering', 'santa costume', 'crossdressing', 'dragon horns', 'legs apart', 'teacup', 'out of frame', 'black wings', 'naughty face', 'demon wings', 'thong', 'cake', 'condom', 'gym uniform', 'short ponytail', 'rape', 'denim shorts', 'curly hair', 'fur collar', 'doggystyle', ':q', 'outside border', 'lolita fashion', 'goggles on head', 'empty eyes', 'alternate breast size', 'science fiction', 'crying with eyes open', "hand on another's head", 'green dress', 'outline', 'buruma', 'jitome', 'light blush', 'futanari', 'monster', 'sunset', 'hands on hips', 'emphasis lines', 'aged up', 'asymmetrical hair', 'high-waist skirt', 'running', 'breast squeeze', 'o o', 'competition swimsuit', '= =', 'angel wings', 'tied hair', 'dog tail', 'minigirl', 'head rest', 'faceless male', '+ +', 'blue theme', 'ghost', 'bespectacled', 'paw pose', 'folded ponytail', 'sarashi', 'popsicle', 'cigarette', 'hand in pocket', 'yukata', 'hand between legs', 'hair down', 'high collar', 'twins', 'dragon girl', 'upside-down', 'serious', 'jeans', 'robe', 'smirk', 'braided ponytail', 'lollipop', ';)', 'hoop earrings', 'dakimakura (medium)', 'bouncing breasts', 'missionary', 'pajamas', '@ @', 'red rose', 'wide-eyed', 'ofuda', 'tentacle hair', 'contemporary', '| |', 'breastplate', ':t', 'blurry foreground', 'threesome', 'hairpin', 'chocolate', 'white hairband', 'jumping', 'sports bra', 'outstretched hand', 'chopsticks', 'oekaki', 'full-face blush', 'bouquet', 'star hair ornament', 'valentine', 'mountain', 'animal hood', 'covering breasts', 'cropped torso', 'wedding dress', 'city', 'video', 'miko', 'glint', 'vibrator', 'animated gif', 'enmaided', 'panties aside', 'doll', 'bloomers', 'backless outfit', 'cum on hair', 'crossed bangs', 'bandaged arm', 'fighting stance', 'hair bell', 'pout', 'multicolored eyes', 'waving', 'colored inner hair', 'raised eyebrows', 'motion blur', 'forehead mark', 'ice cream', 'mouse ears', 'realistic', 'strawberry', 'panties around one leg', 'brother and sister', 'nurse cap', 'hands in pockets', 'center opening', 'leash', 'hair tie', 'clitoris', 'anklet', 'hime cut', 'covering mouth', 'card (medium)', 'dildo', 'sailor hat', 'top-down bottom-up', 'overflow', 'scar across eye', 'sitting on person', 'alternate color', 'mouth mask', 'asymmetrical bangs', 'interracial', 'dragon tail', 'blindfold', 'frog hair ornament', 'adjusting hair', 'spread arms', 'furrowed brow', 'camera', 'licking lips', 'off-shoulder dress', 'wand', 'long dress', 'arms at sides', 'double v', 'arched back', 'zoom layer', 'tall image', 'apple', 'head out of frame', 'hanging breasts', 'sun', 'lowleg', 'sunflower', 'gag', 'labcoat', 'heart hair ornament', 'groping', 'anime screencap', 'yokozuwari', 'alternate hair length', 'bandeau', 'wince', 'gothic lolita', 'shota', 'pasties', 'leg lift', 'dog girl', "jack-o'-lantern", 'highleg panties', 'angel', 'black sclera', 'red hairband', 'partially visible vulva', 'hand on own cheek', 'hand on headwear', 'lactation', 'dot nose', 'hat flower', 'underbust', 'eyeball', 'brown legwear', 'microskirt', 'back-to-back', 'wolf girl', 'ass focus', 'ahegao', 'tabi', 'large areolae', 'food on face', 'nipple slip', 'collared dress', 'criss-cross halter', 'logo', 'pocky', 'character doll', 'gangbang', 'armored dress', 'fairy wings', 'suspender skirt', 'grabbing from behind', 'pointless censoring', 'saliva trail', 'silhouette', "hand on another's shoulder", 'silent comic', 'kneepits', 'onsen', 'crotch seam', 'leaning back', 'gohei', 'butterfly hair ornament', 'holding flower', 'retro artstyle', 'everyone', 'witch', 'no nose', 'guitar', 'center frills', 'cow ears', '2koma', 'sundress', 'pool', 'arm grab', 'heart censor', 'red lips', 'adjusting eyewear', 'bound wrists', 'bangle', 'salute', 'facing away', 'scythe', 'flipped hair', 'classroom', 'waitress', 'chromatic aberration', 'cheerleader', 'strap gap', 'absurdly long hair', 'cleft of venus', 'smug', 'pinafore dress', 'uneven legwear', 'nun', 'food-themed hair ornament', 'inverted nipples', 'hand to own mouth', 'nurse', 'potted plant', 'bikini under clothes', 'road', 'nipple tweak', 'light blue hair', 'ninja', 'sideways glance', 'eyeliner', 'foot focus', 'mecha musume', 'style parody', 'femdom', 'snake hair ornament', 'hair over eyes', 'happy sex', 'mask on head', ':>', '>:)', 'bags under eyes', 'sailor shirt', 'cityscape', 'tube top', 'naked shirt', 'bow bra', 'hair behind ear', 'crown braid', 'bad anatomy', 'brown dress', 'toeless legwear', 'fairy', 'pov hands', 'sarong', 'covering crotch', 'wet shirt', 'pleated dress', 'peach', 'hugging own legs', 'claw pose', 'print legwear', 'lolita hairband', 'bracer', 'indian style', 'alternate eye color', 'own hands clasped', 'dagger', 'painting (medium)', 'hands on own face', 'laughing', 'hair up', 'hibiscus', 'one breast out', 'wet hair', 'heart hands', 'star-shaped pupils', '3d', 'blue rose', 'lace-trimmed legwear', 'alternate hair color', 'akiyama mio', 'carrot', 'drunk', 'pumpkin', 'bodystocking', 'incest', 'marker (medium)', 'stretching', 'crescent hair ornament', 'raccoon ears', 'cone hair bun', 'office lady', 'ribbon choker', 'feather hair ornament', "hand on another's face", 'facepaint', 'red legwear', 'antlers', 'tiger ears', 'pixel art', 'magatama', 'mismatched legwear', 'lace-trimmed panties', 'slippers', 'nipple piercing', 'shibari', 'grey dress', 'wagashi', 'vampire', 'fang out', 'bald', 'bikini skirt', 'highleg bikini', 'hand on own head', 'tachi-e', 'nosebleed', 'evil smile', 'magic circle', 'layered dress', 'princess carry', 'very short hair', 'fishnet legwear', 'legwear under shorts', 'short kimono', 'layered sleeves', 'looking afar', 'hair censor', 'half gloves', 'graphite (medium)', 'scared', 'layered skirt', 'head wreath', 'reference sheet', 'tea', 'one knee', 'yellow dress', 'mother and daughter', 'w', 'open kimono', 'open hand', 'backless dress', 'mermaid', 'lace-trimmed bra', 'greaves', 'limited palette', 'bandaged leg', 'glowing eye', 'field', 'lineart', 'mmf threesome', 'symmetrical docking', 'watercolor (medium)', 'out-of-frame censoring', 'colorized', 'huge ass', 'bedroom', 'breast sucking', 'thumbs up', 'sleeveless turtleneck', 'bound arms', 'halftone', 'undercut', 'naked apron', 'peeing', 'shackles', 'double penetration', 'slingshot swimsuit', 'bra lift', 'idol', 'hair over breasts', 'ruins', 'blue hairband', 'striped dress', 'leggings', '^o^', 'mushroom', 'house', 'dark persona', 'skirt pull', 'purple legwear', 'annoyed', 'fox mask', 'ringed eyes', 'genderswap (ftm)', '3koma', 'arm under breasts', 'face-to-face', 'sad', 'bathroom', 'asymmetrical docking', 'bikini pull', 'bikini armor', 'doughnut', 'yukkuri shiteitte ne', 'babydoll', 'scar on cheek', 'cutoffs', 'hand on own knee', 'headphones around neck', 'two-tone dress', '0 0', 'fundoshi', 'bread', 'sweater dress', 'necktie between breasts', 'white rose', 'cunnilingus', 'x-ray', 'white eyes', 'wide shot', 'big hair', 'covered eyes', 'cow girl', 'constricted pupils', 'casual one-piece swimsuit', 'split', 'anchor hair ornament', 'bobby socks', 'body writing', 'leaf hair ornament', ';o', 'navel piercing', 'no pupils', 'handcuffs', 'untied bikini', 'breast rest', 'yin yang', 'pink theme', 'rolling eyes', 'cookie', 'pink rose', 'spiked collar', 'forehead jewel', 'blue neckwear', 'neck ring', 'gyaru', 'watermelon', 'partially colored', 'bridge', 'purple theme', 'rimless eyewear', 'afterimage', 'folded', 'sheet grab', 'sailor senshi uniform', 'rice', 'badge', 'sitting on lap', 'interspecies', 'lion ears', ':/', 'hooded cloak', 'beer', 'sake', 'heart earrings', 'red theme', 'rainbow', 'egg', 'pointing at viewer', 'giantess', 'd:', 'straight-on', 'mini top hat', 'tying hair', 'feeding', 'lowleg panties', 'pink thighhighs', 'bathtub', 'sepia', 'error', 'cherry', 'torogao', '3:', 'police', 'front ponytail', 'grey legwear', 'race queen', 'faulds', '\\m/', 'pinky out', 'perspective', 'sleepy', 'bear ears', 'seductive smile', 'uwabaki', 'bead bracelet', 'latex', 'hair slicked back', 'reclining', 'taut shirt', 'nervous', 'green theme', 'bead necklace', 'multi-strapped bikini', 'looping animation', 'leaning to the side', 'animalization', 'loose socks', 'blank eyes', 'pink legwear', 'huge ahoge', 'unfinished', 'coffee', 'harness', 'pointy hair', 'fat mons', 'xd', 'hair pulled back', 'blue legwear', 'black neckwear', 'bestiality', 'burger', 'poolside', 'bandaid on leg', 'bra pull', 'multicolored dress', 'nightgown', 'speed lines', 'yawning', 'twisted torso', 'wine', 'whip', 'expressions', 'skull hair ornament', 'standing split', 'braided bun', 'ear blush', 'river', 'gigantic breasts', 'street', 'split mouth', 'castle', 'grapes', 'flat color', 'between fingers', ':|', 'footjob', 'female pervert', 'female ejaculation', 'wet panties', 'police uniform', 'argyle legwear', 'animal collar', 'breast envy', 'sparkling eyes', 'forehead protector', 'checkered', 'long braid', ';p', 'hand on own stomach', 'topknot', 'hand on own ass', 'competition school swimsuit', 'v-neck', 'breast suppress', 'bursting breasts', 'facing another', 'covering face', 'dango', 'costume switch', 'exhibitionism', 'heart cutout', 'yandere', 'gym shirt', 'bokeh', 'sanpaku', 'holding eyewear', 'reaching', 'armpit peek', 'moaning', 'dougi', 'braided bangs', 'panty & stocking with garterbelt', 'fox shadow puppet', 'thong bikini', 'onigiri', 'military jacket', 'aki minoriko', 'hydrangea', 'anal beads', 'stomach bulge', 'fake screenshot', 'noodles', 'leg warmers', 'flower field', 'rose petals', 'breast lift', 'convenient leg', 'jumpsuit', 'solid circle eyes', 'turn pale', 'mind control', 'colored pencil (medium)', 'skyscraper', 'yellow hairband', 'glaring', 'leotard under clothes', 'multiple 4koma', 'heart ahoge', 'chest jewel', 'pink hairband', 'knees to chest', 'double handjob', 'elbow pads', 'kitsune', 'spring onion', 'halter dress', 'pussy peek', 'yellow sclera', 'side braids', 'hooded coat', 'no mouth', 'bra strap', 'naked ribbon', 'shushing', 'impossible shirt', "grabbing another's hair", 'heart-shaped chocolate', 'lowleg bikini', 'ringlets', '>:(', 'detached wings', 'fedora', 'akiyama yukari', 'popped collar', 'dark blue hair', 'pov crotch', 'bow hairband', 'aki shizuha', 'crazy eyes', 'holding strap', 'hands on own knees', 'maebari', 'heart pasties', 'fur-trimmed dress', 'sailor', 'cane', 'ice cream cone', 'business suit', 'tower', 'waterfall', 'unbuttoned shirt', 'bustier', 'butterfly wings', 'dark elf', 'pearl necklace', 'cropped', 'streaming tears', 'torn dress', 'bound legs', 'surgical mask', 'no pussy', 'oppai loli', 'crab', 'leg tattoo', 'kitchen', 'ribbon-trimmed legwear', 'dress pull', 'yellow theme', "hand on another's cheek", 'landscape', 'fusion', 'tsundere', 'cat cutout', 'milk', 'no legwear', 'orange dress', 'long tongue', 'checkered skirt', 'cropped shirt', 'sagging breasts', 'anal tail', 'cat lingerie', 'hatching (texture)', "hand on another's chin", 'cat hair ornament', 'muted color', 'library', '1koma', 'sideways mouth', 'virgin killer sweater', 'mask removed', 'raised eyebrow', 'sleeves pushed up', 'school', 'gradient eyes', 'shy', 'album cover', 'taking picture', 'unaligned breasts', 'mandarin orange', 'colorful', 'clover', 'command spell', 'musical note hair ornament', 'dragon wings', 'backboob', 'carrot hair ornament', 'pointing up', 'husband and wife', 'v over eye', 'half-closed eye', 'barcode', ':>=', 'the pose', 'lake', 'fisheye', 'paizuri under clothes', 'lanyard', 'purple hairband', 'pointing at self', 'dark nipples', 'american flag legwear', 'aerith gainsborough', 'no nipples', 'novelty censor', 'metal collar', 'arm hug', 'cheek-to-cheek', 'shop', 'leaning', 'shrine', 'spider lily', 'nude cover', 'film grain', 'rei no himo', 'banana', 'mother and son', 'green legwear', 'spread ass', 'skirt suit', 'bikini lift', 'pentagram', 'rooftop', 'strapless bikini', 'framed', 'wizard hat', 'chicken', 'pompadour', ';q', 'macaron', 'purple rose', 'trefoil', 'collared jacket', 'slave', 'hair tucking', 'steepled fingers', 'cake slice', 'butt plug', 'swimsuit aside', 'print dress', 'perky breasts', 'pantylines', 'sushi', 'policewoman', 'lower body', 'cream', 'framed breasts', 'official wallpaper', 'egg vibrator', 'heart choker', 'vertical-striped legwear', 'transformation', 'dancer', 'flower-shaped pupils', 'staring', 'bra removed', 'bandaids on nipples', 'frilled legwear', 'lemon', 'capri pants', 'rudder footwear', 'maid bikini', 'hair strand', 'middle finger', 'pizza', 'open dress', 'baozi', 'hair beads', 'candy apple', 'stirrup legwear', 'stage', 'arm belt', 'pussy juice trail', 'chewing gum', 'squirrel ears', 'bandage over one eye', 'gym shorts', 'hairpods', 'meiji schoolgirl uniform', 'breast slip', 'finger gun', 'parfait', 'kindergarten uniform', 'shore', 'drop shadow', 'heart-shaped eyewear', 'gothic', 'multiple hair bows', 'goggles around neck', 'lineup', 'riding crop', 'leotard aside', 'suit jacket', 'upright straddle', 'hair twirling', 'bat hair ornament', 'reverse bunnysuit', 'plaid dress', 'sailor bikini', 'artbook', 'holding underwear', 'bone hair ornament', "hands on another's face", 'sandwich', 'tomato', 'viewfinder', 'layered bikini', 'alternate legwear', 'pancake', 'feather boa', 'pubic hair peek', 'humanization', 'gas mask', 'bamboo forest', 'grinding', 'soda can', 'surreal', 'goggles on headwear', 'bubble tea', 'letterman jacket', 'scowl', 'arms around neck', 'multi-tied hair', 'shooting star', 'lily pad', 'orange hairband', 'magazine cover', 'kanzashi', 'heart in eye', 'raccoon girl', '5koma', 'green neckwear', 'piano', 'snowflake hair ornament', 'crotch rope', 'yellow rose', 'thinking', 'gloom (expression)', 'open-chest sweater', 'left-to-right manga', 'watson cross', 'vertical-striped dress', 'mochi', 'aisaka taiga', 'bodysuit under clothes', 'ribbed dress', 'sheep ears', 'neck ruff', 'lotus', 'breastless clothes', 'chemise', 'cupcake', 'collage', 'covering eyes', 'diffraction spikes']


模型数据 = [
    ('anythingV3_fp16', 'anything-v4.0.vae.pt', 'A3', 'sd', True),
    ('anything-v4.5-pruned-fp32', 'anything-v4.0.vae.pt', 'A4.5', 'sd', False),
    ('AOM3A1', 'orangemix.vae.pt', 'AOM3A1', 'sd', False),
    ('AOM3A2', 'orangemix.vae.pt', 'AOM3A2', 'sd', False),
    ('aoaokoPVCStyleModel_pvcAOAOKO', 'novelailatest-pruned.vae.pt', 'APVC', 'sd', True),
    ('Aidv210AnimeIllustDiffusion_aidv28', 'vae-ft-mse-840000-ema-pruned.ckpt', 'AID28', 'sd', True),
    ('Aidv210AnimeIllustDiffusion_aidv210', 'vae-ft-mse-840000-ema-pruned.ckpt', 'AID210', 'sd', False),
    ('calicomix_v75', None, 'CCM75', 'sd', False),
    ('cosplaymix_v20', None, 'CPM20', 'sd', False),
    ('counterfeitV30_20', 'Counterfeit-V2.5.vae.pt', 'CF2.0', 'sd', True),
    ('Counterfeit-V2.2', 'Counterfeit-V2.5.vae.pt', 'CF2.2', 'sd', True),
    ('Counterfeit-V2.5_pruned', 'Counterfeit-V2.5.vae.pt', 'CF2.5', 'sd', True),
    ('Counterfeit-V3.0_fp16', 'kl-f8-anime2.ckpt', 'CF3.0', 'sd', False),
    ('novelailatest-pruned', 'novelailatest-pruned.vae.pt', 'novelai', 'sd', False),
    ('darkSushiMixMix_225D', 'vae-ft-mse-840000-ema-pruned.ckpt', 'DS225', 'sd', False),
    ('etherBluMix_etherBluMix5', 'kl-f8-anime2.ckpt', 'EB5', 'sd', False),
    ('bluePencil_v9', 'clearvae_v23.safetensors', 'BP9', 'sd', True),
    ('bluePencil_v10', 'clearvae_v23.safetensors', 'BP10', 'sd', False),
    ('AnythingV5Ink_ink', None, 'A5Ink', 'sd', True),
    ('sweetfruit_melon.safetensors_v1.0', 'vae-ft-mse-840000-ema-pruned.ckpt', 'SF1.0', 'sd', False),
    ('cuteyukimixAdorable_midchapter', 'anything-v4.0.vae.pt', 'CYM', 'sd', True),
    ('cuteyukimixAdorable_midchapter2', 'anything-v4.0.vae.pt', 'CYM2', 'sd', True),
    ('cuteyukimixAdorable_midchapter3', 'anything-v4.0.vae.pt', 'CYM3', 'sd', True),
    ('cuteyukimixAdorable_neochapter', 'anything-v4.0.vae.pt', 'CYN', 'sd', True),
    ('cuteyukimixAdorable_neochapter2', 'anything-v4.0.vae.pt', 'CYN2', 'sd', True),
    ('cuteyukimixAdorable_neochapter3', 'anything-v4.0.vae.pt', 'CYN3', 'sd', True),
    ('cuteyukimixAdorable_specialchapter', 'anything-v4.0.vae.pt', 'CYS', 'sd', False),
    ('cuteyukimixAdorable_naiV3style', 'anything-v4.0.vae.pt', 'CYnai3', 'sd', True),
    ('cuteyukimixAdorable_kemiao', 'anything-v4.0.vae.pt', 'CYKM', 'sd', True),
    ('cuteyukimixAdorable_kemiaomiao', 'anything-v4.0.vae.pt', 'CYKMM', 'sd', True),
    ('cuteyukimixAdorable_echodimension', 'anything-v4.0.vae.pt', 'CYE', 'sd', True),
    ('cocotifacute_v20', 'novelailatest-pruned.vae.pt', 'CC20', 'sd', True),
    ('pastelMixStylizedAnime_pastelMixPrunedFP16', 'kl-f8-anime2.ckpt', 'PM', 'sd', False),
    ('perfectWorld_v2Baked', None, 'PW2', 'sd', True),
    ('perfectWorld_v6Baked', None, 'PW6', 'sd', False),
    ('himawarimix_v100', None, 'HW100', 'sd', False),
    ('meinamix_meinaV11', None, 'MM11', 'sd', False),
    ('mixProV4_v4', 'novelailatest-pruned.vae.pt', 'MP4', 'sd', False),
    ('cetusMix_cetusVersion3', 'kl-f8-anime2.ckpt', 'CM3', 'sd', True),
    ('cetusMix_Coda2', 'kl-f8-anime2.ckpt', 'CMC2', 'sd', True),
    ('cetusMix_Whalefall2', 'kl-f8-anime2.ckpt', 'CMWF2', 'sd', True),
    ('cetusMix_v4', 'kl-f8-anime2.ckpt', 'CM4', 'sd', False),
    ('cetusMix_cetusVersion2', 'kl-f8-anime2.ckpt', 'CM2', 'sd', True),
    ('sakuramochimix_v10', 'novelailatest-pruned.vae.pt', 'SMM10', 'sd', False),
    ('sweetMix_v22Flat', 'blessed2.vae.safetensors', 'SM22F', 'sd', True),
    ('ghostmix_v20Bakedvae', None, 'GM20', 'sd', False),
    ('anyloraCheckpoint_novaeFp16', 'kl-f8-anime2.ckpt', 'AL', 'sd', False),
    ('PVCStyleModelMovable_v20NoVae', 'vae-ft-mse-840000-ema-pruned.ckpt', 'PVC20', 'sd', False),
    ('PVCStyleModelMovable_v30', 'vae-ft-mse-840000-ema-pruned.ckpt', 'PVC30', 'sd', True),
    ('divineelegancemix_V9', 'MoistMix.vae.pt', 'DLM9', 'sd', False),
    ('rainbowsweets_v20', None, 'RS20', 'sd', False),
    ('rabbit_v7', 'novelailatest-pruned.vae.pt', 'R7', 'sd', True),
    ('rimochan_random_mix', 'blessed2.vae.safetensors', 'RRM', 'sd', True),
    ('rimochan_random_mix_1.1', 'blessed2.vae.safetensors', 'RRM1.1', 'sd', True),
    ('rimochan_random_mix_2.1', 'blessed2.vae.safetensors', 'RRM2.1', 'sd', True),
    ('rimochan_random_mix_3.2', 'blessed2.vae.safetensors', 'RRM3.2', 'sd', True),
    ('rimochan_random_mix_4.0', 'blessed2.vae.safetensors', 'RRM4.0', 'sd', True),
    ('rimochan_random_mix_4.1', 'blessed2.vae.safetensors', 'RRM4.1', 'sd', False),
    ('koji_v21', 'clearvae_v23.safetensors', 'KJ21', 'sd', False),
    ('kaywaii_v50', 'clearvae_v23.safetensors', 'KW50', 'sd', True),
    ('kaywaii_v60', 'clearvae_v23.safetensors', 'KW60', 'sd', True),
    ('kaywaii_v70', 'clearvae_v23.safetensors', 'KW70', 'sd', True),
    ('kaywaii_v80', 'clearvae_v23.safetensors', 'KW80', 'sd', True),
    ('kaywaii_v85', 'clearvae_v23.safetensors', 'KW85', 'sd', True),
    ('kaywaii_v90', 'clearvae_v23.safetensors', 'KW90', 'sd', False),
    ('jitq_v20', 'blessed2.vae.safetensors', 'JQ20', 'sd', False),
    ('jitq_v30', 'blessed2.vae.safetensors', 'JQ30', 'sd', True),
    ('petitcutie_v15', 'blessed2.vae.safetensors', 'PC15', 'sd', True),
    ('petitcutie_v20', 'blessed2.vae.safetensors', 'PC20', 'sd', True),
    ('superInvincibleAnd_v2', 'blessed2.vae.safetensors', 'SIA2', 'sd', False),
    ('ApricotEyes_v10', 'blessed2.vae.safetensors', 'ACE10', 'sd', True),
    ('aiceKawaice_channel', 'blessed2.vae.safetensors', 'AKC', 'sd', True),
    ('coharumix_v6', 'blessed2.vae.safetensors', 'CHM6', 'sd', False),
    ('irismix_v90', None, 'I90', 'sd', True),
    ('yetanotheranimemodel_v20', 'blessed2.vae.safetensors', 'YAA20', 'sd', True),
    ('theWondermix_v12', 'blessed2.vae.safetensors', 'TWM12', 'sd', False),
    ('animeIllustDiffusion_v052', 'sdxl_vae.safetensors', 'AIDXL52', 'sdxl', False),
    ('animeIllustDiffusion_v061', 'sdxl_vae.safetensors', 'AIDXL61', 'sdxl', True),
    ('CounterfeitXL-V1.0', None, 'CFXL1.0', 'sdxl', True),
    ('counterfeitxl_v20', None, 'CFXL2.0', 'sdxl', True),
    ('counterfeitxl_v25', None, 'CFXL2.5', 'sdxl', False),
    ('hassakuXLSfwNsfwBeta_betaV01', None, 'HXLB01', 'sdxl', True),
    ('hassakuXLHentai_v13', None, 'HXLHv13', 'sdxl', False),
    ('reproductionSDXL_2v12', None, 'RXL2v12', 'sdxl', False),
    ('kohakuXLBeta_beta7', 'sdxl_vae.safetensors', 'KXLB7', 'sdxl', False),
    ('kohakuXLDelta_rev1', 'sdxl_vae.safetensors', 'KXLD1', 'sdxl', True),
    ('kohakuXLEpsilon_rev1', 'sdxl_vae.safetensors', 'KXLE1', 'sdxl', False),
    ('blue_pencil-XL-v0.3.1', None, 'BPXL0.3.1', 'sdxl', True),
    ('bluePencilXL_v200', None, 'BPXL2.0.0', 'sdxl', True),
    ('bluePencilXL_v600', None, 'BPXL6.0.0', 'sdxl', False),
    ('himawarimix_xlV13', None, 'HWXL13', 'sdxl', False),
    ('animagineXLV3_v30', None, 'AMXL30', 'sdxl', False),
    ('aniease_v24', None, 'AE24', 'sdxl', False),
    ('cutecore_xl', None, 'CCXL', 'sdxl', False),
    ('aingdiffusionXL_v06', None, 'ADXL06', 'sdxl', True),
    ('aingdiffusionXL_v11', None, 'ADXL11', 'sdxl', False),
    ('ConfusionXL1.0', 'sdxl_vae_0.9.safetensors', 'CXL1.0', 'sdxl', True),
    ('ConfusionXL2.0', 'sdxl_vae_0.9.safetensors', 'CXL2.0', 'sdxl', True),
    ('ConfusionXL3.0', 'sdxl_vae_0.9.safetensors', 'CXL3.0', 'sdxl', True),
    ('ConfusionXL4.0_fp16_vae', None, 'CXL4.0', 'sdxl', True),
    ('ConfusionXL5.0', 'sdxl_vae_0.9.safetensors', 'CXL5.0', 'sdxl', True),
    ('ConfusionXL5.0B', 'sdxl_vae_0.9.safetensors', 'CXL5.0B', 'sdxl', False),
    ('PVCStyleModelFantasy_betaV10', 'sdxl_vae.safetensors', 'PVCFB10', 'sdxl', True),
    ('PVCStyleModelMovable_beta25Realistic', None, 'PVCFB25', 'sdxl', True),
    ('PVCStyleModelMovable_pony160', None, 'PVCP160', 'sdxl', False),
    ('netaArtXL_v10', None, 'NAXL10', 'sdxl', False),
    ('illustriousXL_v01', None, 'IXL0.1', 'sdxl', False),
    ('noobaiXLNAIXL_earlyAccessVersion', None, 'NBE0.25', 'sdxl', True),
    ('noobaiXLNAIXL_epsilonPred05Version', None, 'NBE0.5', 'sdxl', True),
    ('noobaiXLNAIXL_epsilonPred075', None, 'NBE0.75', 'sdxl', True),
    ('noobaiXLNAIXL_epsilonPred10Version', None, 'NBE1.0', 'sdxl', False),
    ('noobaiXLNAIXL_epsilonPred11Version', None, 'NBE1.1', 'sdxl', False),
    ('noobaiXLNAIXL_vPred05Version', None, 'NBV0.5', 'sdxl', True),
    ('noobaiXLNAIXL_vPred06Version', None, 'NBV0.6', 'sdxl', False),
    ('waiNSFWIllustrious_v110', None, 'WNI11.0', 'sdxl', False),
    ('noobaiXL_XYZ', None, 'NBX', 'sdxl', True),
    ('noobaiXL_CXyz_2', None, 'NBCX', 'sdxl', False),
    ('pdForAnime_v20', None, 'PDFA20', 'sdxl', False),
    ('ponyDiffusionV6XL_v6StartWithThisOne', None, 'P6XL', 'sdxl', False),
    ('tPonynai3_v6', None, 'TPnai3v6', 'sdxl', False),
    ('thickCoatingStyle_pdxl10', None, 'TCPD10', 'sdxl', True),
    ('flux1-schnell-fp8', None, 'FLUX1S', 'flux.1s', False),
    ('flux1-dev-fp8', None, 'FLUX1D', 'flux.1d', False),
    ('nai-diffusion-3', None, 'NAI3', 'nai3', False),
]


人频率1 = {k: v for k, v in orjson.loads(open(此处 / 'data\人频率6000000~7000000.json').read()).items() if v > 24}
人频率2 = {k: v for k, v in orjson.loads(open(此处 / 'data\人频率1~6400000.json').read()).items() if v > 64}
要测的人 = sorted(set(人频率1) | set(人频率2))
