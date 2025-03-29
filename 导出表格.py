import re
import copy
from pathlib import Path

import orjson
import numpy as np
import pandas as pd
from tqdm import tqdm
from common import 模型数据, 要测的人


Path('测试结果').mkdir(exist_ok=True)


模型数据d = {i[0]: i for i in 模型数据}


def 获取模型(x: str):
    if t := 模型数据d.get(x):
        return t
    return [x, None, x, 'sd', False]


def _is_XL(x):
    return {i[2]: i[3]=='sdxl' for i in 模型数据}.get(x, False)


readme_mode = False
readme要的 = {'animagineXL40_v4Opt', 'illustriousXL10_v10', 'ConfusionXL5.0B', 'noobaiXLNAIXL_epsilonPred10Version', 'nai-diffusion-3', 'waiNSFWIllustrious_v120'}


def _加粗(data: dict[str, list], yy):
    data = copy.deepcopy(data)
    xx = [*data.keys()]
    for i, _ in enumerate(yy):
        top = sorted([data[x][i] for x in xx], reverse=True, key=lambda x: x if isinstance(x, float|int) else -1)[:3]
        for x in xx:
            if data[x][i] in top and data[x][i] != '-':
                data[x][i] = f'**{data[x][i]}**'
    return data


def _分离(x: list[float], y: list[float], t=0.002, iter=3):
    x = copy.deepcopy(x)
    y = copy.deepcopy(y)
    for _ in range(iter):
        for i in range(len(x)):
            for j in range(len(y)):
                if i == j:
                    continue
                if (((x[i]-x[j])/3)**2+(y[i]-y[j])**2)**0.5 < t:
                    x[i] -= (-1)**(x[i] > x[j]) * (t/10)
                    x[j] += (-1)**(x[i] > x[j]) * (t/10)
                    y[i] -= (-1)**(y[i] > y[j]) * (t/10)
                    y[j] += (-1)**(y[i] > y[j]) * (t/10)
    return x, y


def 导出单标签(width=512):
    m = {}
    all_model = set()
    all_tag = set()
    for 文件 in tqdm([*Path('savedata').glob('单标签_*_记录.json')]):
        for d in orjson.loads(open(文件, 'rb').read()):
            if d['参数']['width'] != width:
                continue
            sd_model_checkpoint = d['参数']['override_settings']['sd_model_checkpoint']
            assert sd_model_checkpoint in str(文件)
            model = sd_model_checkpoint
            好 = len([i for i in d['分数'] if i > 0.1])
            n = len(d['分数'])
            assert (model, d['标签']) not in m
            m[model, d['标签']] = 好, n
            all_model.add(model)
            all_tag.add(d['标签'])
    all_model = sorted(all_model, key=lambda x: x if _is_XL(x) else '0' + x)
    all_tag = sorted(all_tag)

    满标签 = []     # 全部不为None
    好标签 = sorted(all_tag)
    for tag in 好标签:
        for model in all_model:
            if m.get((model, tag)) is None:
                break
        else:
            满标签.append(tag)

    data = {}
    for model in all_model:
        data[model] = []
        for tag in 好标签:
            t = m.get((model, tag))
            if t is None:
                data[model].append('-')
            else:
                好, n = t
                data[model].append(好 / n)
    if readme_mode:
        data = {k: v for k, v in data.items() if k in readme要的}
    df = pd.DataFrame(data, index=好标签)
    df.to_pickle(f'测试结果/模型对单标签-准确率{readme_mode or ""}.pkl')
    with open(f'测试结果/模型对单标签-准确率{readme_mode or ""}.md', 'w', encoding='utf8') as f:
        f.write('# 模型对单标签-准确率: \n\n<sub>\n\n' + df.to_markdown() + '\n\n</sub>\n\n')

    目录 = orjson.loads(open('data/目录.json', encoding='utf-8').read())
    目录['总体'] = {'name': '总体', 'keys': []}
    逆转目录 = {}
    for k, v in 目录.items():
        for i in v['keys']:
            逆转目录[i.lower().replace(' ', '_')] = k
    超逆转目录 = {k: v.split('-')[0] for k, v in 逆转目录.items()}
    for 逆, 文件名 in [(逆转目录, '模型对标签类别-准确率'), (超逆转目录, '模型对标签大类-准确率')]:
        mm = {}
        for (model, tag), (好, n) in m.items():
            if tag not in 满标签:
                continue
            for 大 in [逆[tag], '总体']:
                原好, 原n = mm.get((model, 大), (0, 0))
                mm[model, 大] = 好 + 原好, n + 原n
        data = {}
        sorted_目录 = sorted({k[1] for k in mm})
        for model in all_model:
            data[model] = []
            for 大 in sorted_目录:
                t = mm.get((model, 大))
                if t is None:
                    data[model].append('-')
                else:
                    好, n = t
                    if n <= 32:     # 不置信
                        data[model].append('-')
                    else:
                        data[model].append(round(好 / n, 3))
        if readme_mode:
            data = {k: v for k, v in data.items() if k in readme要的}
        pd.DataFrame(data, index=[目录.get(i, {}).get('name', i) for i in sorted_目录]).to_pickle(f'测试结果/{文件名}.pkl')
        df = pd.DataFrame(data, index=[目录.get(i, {}).get('name', i) for i in sorted_目录])
        with open(f'测试结果/{文件名}{width}{readme_mode or ""}.md', 'w', encoding='utf8') as f:
            f.write(f'# {文件名}: \n\n<sub>\n\n' + df.to_markdown() + '\n\n</sub>\n\n')


def 导出单标签2():
    标签计数: dict[str, int] = {}
    模型标签计数: dict[str, dict[str, int]] = {}
    for 文件 in tqdm([*Path('savedata').glob('单标签_*_记录.json')]):
        for d in orjson.loads(open(文件, 'rb').read()):
            model = d['参数']['override_settings']['sd_model_checkpoint']
            计 = 模型标签计数.setdefault(model, {})
            for v in d['预测标签'].values():
                for kk, vv in v.items():
                    标签计数.setdefault(kk, 0)
                    标签计数[kk] += vv
                    计.setdefault(kk, 0)
                    计[kk] += vv
    n = len(模型标签计数)
    data = {}
    top_n = 12
    for k, v in sorted(模型标签计数.items()):
        差v = {kk: (vv / (标签计数.get(kk)/n + 1000)) for kk, vv in v.items()}
        data[k] = [x[0] for x in sorted(差v.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    df = pd.DataFrame(data, index=[f'top_{i}' for i in [*range(1, top_n+1)]])
    with open('测试结果/模型偏好标签.md', 'w', encoding='utf8') as f:
        f.write('# 模型偏好标签: \n' + df.to_markdown() + '\n\n')
    breasts = {
        'flat_chest': 0,
        'small_breasts': 0.2,
        'medium_breasts': 0.4,
        'large_breasts': 0.6,
        'huge_breasts': 0.8,
        'gigantic_breasts': 1,
    }
    hair = {
        'short_hair': 0,
        'medium_hair': 0.25,
        'long_hair': 0.5,
        'very_long_hair': 0.75,
        'absurdly_long_hair': 1,
    }
    头发颜色 = ['blue_hair', 'red_hair', 'pink_hair', 'purple_hair', 'brown_hair', 'orange_hair', 'black_hair', 'blonde_hair', 'dark_blue_hair', 'light_purple_hair', 'light_brown_hair', 'white_hair', 'silver_hair', 'grey_hair', 'light_blue_hair', 'green_hair']
    q = {}
    for k, v in sorted(模型标签计数.items()):
        for 种类, d in [('胸部大小', breasts), ('头发长度', hair)]:
            总个数 = 0
            总分数 = 0
            for h, hv in d.items():
                总个数 += v.get(h, 0)
                总分数 += v.get(h, 0) * hv
            q.setdefault(k, {}).setdefault(种类, 总分数/总个数)
        q[k]['头发颜色'] = max([(v.get(x, 0), x) for x in 头发颜色])[1].removesuffix('_hair').replace('blonde', 'yellow')
    with open('测试结果/模型偏好角色属性.md', 'w', encoding='utf8') as f:
        f.write('# 模型偏好角色属性: \n\n<sub>\n\n' + pd.DataFrame(q).T.to_markdown() + '\n\n</sub>\n\n')
    from bokeh.plotting import figure, show, save
    from bokeh.models.annotations import Label
    模型 = [*模型标签计数]
    x = [q[i]['胸部大小'] for i in 模型]
    y = [q[i]['头发长度'] for i in 模型]
    x, y = _分离(x, y, t=0.004)
    color = [q[i]['头发颜色'] for i in 模型]
    p = figure(title="散点图", x_axis_label="胸部大小", y_axis_label="头发长度", x_range = (min(x)-0.005, max(x)+0.03), width=1440, height=880)
    p.circle(x, y, size=10, color=color)
    for i in range(len(x)):
        label = Label(x=x[i]+0.0014, y=y[i]-0.0046, text=模型[i], text_font_size='9pt')
        p.add_layout(label)
    save(p, '导出单标签2.html')


def 导出多标签(width=512):
    m = {}
    for 文件 in tqdm([*Path('savedata').glob('多标签_*_记录.json')]):
        for d in orjson.loads(open(文件, 'rb').read()):
            n = len(d['标签组'])
            if d['参数']['width'] != width:
                continue
            model = d['参数']['override_settings']['sd_model_checkpoint']
            m.setdefault((model, n), {'相似度': [], '分数': []})
            m[model, n]['相似度'].extend(d['相似度'])
            m[model, n]['分数'].extend(d['分数'])
    print(d['参数']['width'])
    all_model, all_n = zip(*m.keys())
    all_model = sorted({*all_model}, key=lambda x: x if _is_XL(x) else '0' + x)
    all_n = sorted({*all_n})

    data = {}
    data2 = {}
    d_shape = {}
    for model in all_model:
        data[model] = []
        data2[model] = []
        for n in all_n:
            t = m.get((model, n))
            if t is None:
                data[model].append('-')
                data2[model].append('-')
            else:
                a = np.array(m[model, n]['分数'])
                assert d_shape.setdefault(a.shape[0], a.shape) == a.shape, f'{model}测试结果的形状{a.shape}不对！'
                acc = (a > 0.001).sum() / len(a.flatten())
                data[model].append(round(acc, 3))
                data2[model].append(round(1 - np.array(m[model, n]['相似度']).mean(), 3))
    if readme_mode:
        data = {k: v for k, v in data.items() if k in readme要的}
        data2 = {k: v for k, v in data2.items() if k in readme要的}
    with open(f'测试结果/模型对标签个数-准确率和多样性{readme_mode or ""}{str(width)*(width!=512)}.md', 'w', encoding='utf8') as f:
        f.write('# 模型对标签个数-准确率: \n\n<sub>\n\n' + pd.DataFrame(data, index=all_n).to_markdown() + '\n\n</sub>\n\n')
        f.write('# 模型对标签个数-多样性: \n\n<sub>\n\n' + pd.DataFrame(data2, index=all_n).to_markdown() + '\n\n</sub>\n\n')

    if not readme_mode:
        from bokeh.plotting import figure, save
        from bokeh.models.annotations import Label
        d03 = {i[0]: i[3] for i in 模型数据}
        d04 = {i[0]: i[4] for i in 模型数据}
        好all_model = [i for i in all_model if not d04.get(i)]
        x = [data[i][all_n.index(32)] for i in 好all_model]
        y = [data2[i][all_n.index(32)] for i in 好all_model]
        for i, v in enumerate(x):
            if v == '-':
                x[i] = 0
        for i, v in enumerate(y):
            if v == '-':
                y[i] = 0
        x, y = _分离(x, y)
        p = figure(title="散点图", x_axis_label="准确度", y_axis_label="多样性", x_range = (min(x)-0.005, max(x)+0.037), width=1440, height=880)
        color_map = {
            'sd': 'blue',
            'sdxl': 'red',
            'nai3': 'red',
            'flux.1s': 'green',
            'flux.1d': 'green',
        }
        color = [color_map[d03.get(i, 'sd')] for i in 好all_model]
        p.circle(x, y, size=10, color=color, alpha=0.5)
        for i in range(len(x)):
            label = Label(x=x[i]+0.0016, y=y[i]-0.0011, text=好all_model[i], text_font_size='8pt')
            p.add_layout(label)
        save(p, f'导出多标签{str(width)*(width!=512)}.html')


def 导出不同参数():
    l = orjson.loads(open('savedata/记录_不同参数.json', encoding='utf-8').read())
    d = {}
    for i in l:
        model = i['参数']['override_settings']['sd_model_checkpoint']
        参数 = i['参数']
        x = f'{参数["width"]}×{参数["height"]}'
        y = 参数['steps']
        d.setdefault((model, x, y), []).extend(i['分数'])
    dd = {}
    for (model, x, y), 分数 in d.items(): 
        a = np.array(分数)
        acc = (a > 0.001).sum() / len(a.flatten())
        dd.setdefault(model, {})[x, y] = acc
    
    with open('测试结果/不同模型在不同尺寸和step下的准确率.md', 'w', encoding='utf8') as f:
        f.write('# 不同模型在不同尺寸和step下的准确率: \n\n')
        for model in dd:
            all_x, all_y = map(sorted, map(set, zip(*dd[model].keys())))
            data = {}
            for x in all_x:
                data[x] = []
                for y in all_y:
                    data[x].append(dd[model][x, y])
            f.write(f'## {model}\n')
            f.write(f'{pd.DataFrame(data, index=all_y).to_markdown()}\n\n')


def 导出角色():
    作品人 = {
        'touhou': ['alice margatroid', 'chen', 'cirno', 'clownpiece', 'daiyousei', 'flandre scarlet', 'fujiwara no mokou', 'futatsuiwa mamizou', 'hakurei reimu', 'hata no kokoro', 'hieda no akyuu', 'hinanawi tenshi', 'hong meiling', 'houjuu nue', 'ibaraki kasen', 'imaizumi kagerou', 'inaba tewi', 'inubashiri momiji', 'izayoi sakuya', 'kaenbyou rin', 'kaku seiga', 'kamishirasawa keine', 'kasodani kyouko', 'kawashiro nitori', 'kazami yuuka', 'kijin seija', 'kirisame marisa', 'kishin sagume', 'koakuma', 'kochiya sanae', 'komeiji koishi', 'komeiji satori', 'konpaku youmu', 'letty whiterock', 'mizuhashi parsee', 'moriya suwako', 'murasa minamitsu', 'mystia lorelei', 'nagae iku', 'nazrin', 'onozuka komachi', 'patchouli knowledge', 'reisen udongein inaba', 'reiuji utsuho', 'remilia scarlet', 'rumia', 'saigyouji yuyuko', 'sekibanki', 'shameimaru aya', 'tatara kogasa', 'toramaru shou', 'toyosatomimi no miko', 'usami renko', 'yakumo ran', 'yakumo yukari', 'hijiri byakuren', 'shiki eiki','kagiyama hina', 'ibuki suika','houraisan kaguya', 'himekaidou hatate', 'mononobe no futo', 'wakasagihime','wriggle nightbug', 'hecatia lapislazuli', 'doremy sweet','yorigami shion', "yorigami jo'on", 'sukuna shinmyoumaru', 'kurodani yamame', 'usami sumireko', 'yagokoro eirin', 'soga no tojiko', 'miyako yoshika', 'yasaka kanako', 'motoori kosuzu', 'hyakumantenbara salome'],
        'idolmaster': ['amami haruka', 'ganaha hibiki', 'hoshii miki', 'jougasaki mika', 'maekawa miku', 'minase iori', 'shijou takane', 'takagaki kaede', 'yumemi riamu', 'yumemi riamu', 'shibuya rin', 'ichinose shiki', 'sagisawa fumika', 'higuchi madoka', 'kanzaki ranko', 'shirasaka koume', 'mayuzumi fuyuko', 'hayami kanade', 'kokubunji suou', 'hachimiya meguru', 'koshimizu sachiko', 'nitta minami', 'sunazuka akira', 'shirase sakuya', 'futaba anzu', 'shimamura uzuki', 'konoe kanata', 'hoshi syoko', 'morino rinze', 'jougasaki rika', 'kamiya nao', 'akagi miria', 'asakura toru', 'ohtsuki yui'],
        'fate': ['mash kyrielight', 'illyasviel von einzbern', 'tohsaka rin', 'matou sakura', 'miyu edelfelt', 'chloe von einzbern'],
        'genshin_impact': ['raiden shogun', 'yae miko', 'sangonomiya kokomi', 'kamisato ayaka'],
        'hololive': ['houshou marine', 'gawr gura', 'minato aqua', 'shirakami fubuki', 'kitagawa marin', 'hoshimachi suisei', 'uruha rushia', "ninomae ina'nis", 'gotou hitori', 'nakiri ayame', 'nekomata okayu', 'shirogane noel', 'usada pekora', 'amane kanata', 'ookami mio', 'inugami korone', 'murasaki shion', 'sakura miko', 'tokoyami towa', 'yukihana lamy', 'akai haato', 'natsuiro matsuri', 'hoshikawa sara', 'tsukino mito', 'oozora subaru', 'kiryu coco', 'shishiro botan']
    }
    作品名 = {
        'touhou': '东方',
        'genshin_impact': '原神',
        'fate': 'fate',
        'hololive': 'hololive',
        'kancolle': '舰娘',
        'idolmaster': '偶像大师',
        'blue_archive': '蔚蓝档案',
    }
    def _人对应作品(x):
        a = re.findall('.+\((.+)\)', x)
        if a:
            return a[0]
        for k, v in 作品人.items():
            if x.replace('_', ' ') in v:
                return k
    data = {}
    全m = {}
    for 文件 in [*Path('savedata').glob('人物_*_记录.json')]:
        with open(文件, 'r', encoding='utf8') as f:
            记录 = orjson.loads(f.read())
        m = {}
        for d in 记录:
            if d['人'].replace('_', ' ') not in 要测的人:
                continue
            model = d['参数']['override_settings']['sd_model_checkpoint']
            m[d['人']] = np.mean([d['人'] in i for i in d['预测']])
        全m[model] = m
        z = {k: [0, 0] for k in sorted(作品名)}
        for 人 in m:
            if 作品 := _人对应作品(人):
                z.setdefault(作品, [0, 0])
                z[作品][0] += m[人]
                z[作品][1] += 1
        data[model] = [f'{int(z[k][0])}/{z[k][1]}' for k in sorted(作品名)]
        data[model].append(f'{int(np.sum([*m.values()]))}/{len(m)}')
    if readme_mode:
        data = {k: v for k, v in sorted(data.items(), key=lambda x: x[0] if _is_XL(x[0]) else '0' + x[0]) if k in readme要的}
    with open(f'测试结果/模型对不同作品的准确率{readme_mode or ""}.md', 'w', encoding='utf8') as f:
        f.write(f'{pd.DataFrame(data, index=[作品名[k] for k in sorted(作品名)]+["总体"]).to_markdown()}\n\n')
    df = pd.DataFrame(全m)
    df.to_pickle('测试结果/模型对不同角色的准确率.pkl')
    with open('测试结果/模型对不同角色的准确率.md', 'w', encoding='utf8') as f:
        f.write(f'{df.to_markdown()}\n\n')


def 导出lvis():
    d = {}
    for 文件 in tqdm([*Path('savedata').glob('lvis_*_记录.json')]):
        for dd in orjson.loads(open(文件, 'rb').read()):
            model = dd['参数']['override_settings']['sd_model_checkpoint']
            n = len(dd['标签组'])
            分数 = []
            for names, scores in dd['检测结果']:
                ddd = {}
                for name, score in zip(names, scores):
                    ddd[name] = max(ddd.get('name', 0), score)
                分数.append(len(set(names))/n)
            d.setdefault(model, []).extend(分数)
    data = {model: sum(z)/len(z) for model, z in d.items()}
    if readme_mode:
        data = {k: v for k, v in data.items() if k in readme要的}
    with open(f'测试结果/不同模型在lvis下的准确率{readme_mode or ""}.md', 'w', encoding='utf8') as f:
        f.write(f'{pd.DataFrame(data, index=["score"]).to_markdown()}\n\n')


if __name__ == '__main__':
    导出单标签(512)
    导出单标签(768)
    导出单标签2()
    导出多标签()
    导出多标签(768)
    导出多标签(1024)
    导出角色()
    导出lvis()

    readme_mode = True
    导出单标签(512)
    导出单标签(768)
    导出多标签(768)
    导出角色()
    导出lvis()
