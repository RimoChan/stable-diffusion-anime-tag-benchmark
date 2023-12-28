import copy

import orjson
import numpy as np
import pandas as pd


output = open('好.md', 'w', encoding='utf8')


def _模型改名(x):    # 为了让表格在 GitHub 上显示更好看
    return {
        'AnythingV5Ink_ink': 'A5Ink',
        'anything-v4.5-pruned-fp32': 'A4.5',
        'Counterfeit-V2.2': 'CF2.2',
        'Counterfeit-V3.0_fp16': 'CF3.0',
        'counterfeitV30_20': 'CF2.0',
        'cosplaymix_v20': 'CPM20',
        'novelailatest-pruned': 'novelai',
        'sweetfruit_melon.safetensors_v1.0': 'SF1.0',
        'bluePencil_v9': 'BP9',
        'bluePencil_v10': 'BP10',
        'blue_pencil-XL-v0.3.1': 'BPXL0.3.1',
        'CounterfeitXL-V1.0': 'CFXL1.0',
        'counterfeitxl_v20': 'CFXL2.0',
        'counterfeitxl_v25': 'CFXL2.5',
        'cuteyukimixAdorable_midchapter2': 'CYM2',
        'cuteyukimixAdorable_midchapter3': 'CYM3',
        'cuteyukimixAdorable_neochapter3': 'CYN3',
        'cuteyukimixAdorable_specialchapter': 'CYS',
        'cuteyukimixAdorable_naiV3style': 'CYnai3',
        'cuteyukimixAdorable_kemiao': 'CYKM',
        'cuteyukimixAdorable_kemiaomiao': 'CYKMM',
        'Counterfeit-V2.5_pruned': 'CF2.5',
        'cocotifacute_v20': 'CC20',
        'etherBluMix_etherBluMix5': 'EB5',
        'perfectWorld_v2Baked': 'PW2',
        'perfectWorld_v6Baked': 'PW6',
        'meinamix_meinaV11': 'MM11',
        'mixProV4_v4': 'MP4',
        'cetusMix_cetusVersion3': 'CM3',
        'cetusMix_v4': 'CM4',
        'cetusMix_Whalefall2': 'CMWF2',
        'cetusMix_Coda2': 'CMC2',
        'sakuramochimix_v10': 'SM10',
        'anyloraCheckpoint_novaeFp16': 'AL',
        'anythingV3_fp16': 'A3',
        'ghostmix_v20Bakedvae': 'GM20',
        'aoaokoPVCStyleModel_pvcAOAOKO': 'APVC',
        'PVCStyleModelMovable_v20NoVae': 'PVC20',
        'divineelegancemix_V9': 'DLM9',
        'darkSushiMixMix_225D': 'DS225',
        'koji_v21': 'KJ21',
        'kaywaii_v50': 'KW50',
        'kaywaii_v60': 'KW60',
        'kaywaii_v70': 'KW70',
        'kaywaii_v80': 'KW80',
        'rainbowsweets_v20': 'RS20',
        'rabbit_v7': 'R7',
    }.get(x, x)


readme要的 = {'A5Ink', 'AL', 'AOM3A1', 'BP10', 'CF3.0', 'CM4', 'CYS', 'KW80', 'MM11', 'SF1.0', 'SM10', 'novelai', 'BPXL0.3.1', 'CFXL2.5'}


def _加粗(data: dict[str, list], yy):
    data = copy.deepcopy(data)
    xx = [*data.keys()]
    for i, _ in enumerate(yy):
        top = sorted([data[x][i] for x in xx], reverse=True, key=lambda x: x if isinstance(x, float|int) else -1)[:3]
        for x in xx:
            if data[x][i] in top and data[x][i] != '-':
                data[x][i] = f'**{data[x][i]}**'
    return data


def _分离(x: list[float], y: list[float], t=0.002, iter=2):
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


def 导出单标签():
    l = orjson.loads(open('savedata/记录.json', encoding='utf-8').read())
    m = {}
    all_model = set()
    all_tag = set()
    for d in l:
        model = _模型改名(d['参数']['override_settings']['sd_model_checkpoint'])
        好 = len([i for i in d['分数'] if i > 0.1])
        n = len(d['分数'])
        assert (model, d['标签']) not in m
        m[model, d['标签']] = 好, n
        all_model.add(model)
        all_tag.add(d['标签'])
    all_model = sorted(all_model, key=lambda x: x if 'XL' in x else '0' + x)
    all_tag = sorted(all_tag)

    好标签 = []     # 至少1个不为0且不为None
    满标签 = []     # 至少1个不为0，全部不为None
    for (model, tag), (好, n) in m.items():
        if 好 > 1:
            好标签.append(tag)
    好标签 = sorted({*好标签})
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
    df = pd.DataFrame(data, index=好标签)
    output.write('# 模型对单标签-准确率: \n' + df.to_markdown() + '\n\n')

    目录 = orjson.loads(open('data/目录.json', encoding='utf-8').read())
    逆转目录 = {}
    mm = {}
    for k, v in 目录.items():
        for i in v['keys']:
            逆转目录[i.lower().replace(' ', '_')] = k
    for (model, tag), (好, n) in m.items():
        if tag not in 满标签:
            continue
        大 = 逆转目录[tag]
        原好, 原n = mm.get((model, 大), (0, 0))
        mm[model, 大] = 好 + 原好, n + 原n
    data = {}
    sorted_目录 = sorted(目录)
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
    # data = {k: v for k, v in data.items() if k in readme要的}
    df = pd.DataFrame(_加粗(data, sorted_目录), index=[目录[i]['name'] for i in sorted_目录])
    output.write('# 模型对标签类别-准确率: \n' + df.to_markdown() + '\n\n')


def 导出单标签2():
    l = orjson.loads(open('savedata/记录.json', encoding='utf-8').read())
    标签计数: dict[str, int] = {}
    模型标签计数: dict[str, dict[str, int]] = {}
    for d in l:
        model = _模型改名(d['参数']['override_settings']['sd_model_checkpoint'])
        计 = 模型标签计数.setdefault(model, {})
        for v in d['预测标签'].values():
            for kk, vv in v.items():
                标签计数.setdefault(kk, 0)
                标签计数[kk] += vv
                计.setdefault(kk, 0)
                计[kk] += vv
    n = len(模型标签计数)
    data = {}
    for k, v in sorted(模型标签计数.items()):
        差v = {kk: (vv / (标签计数.get(kk)/n + 1000)) for kk, vv in v.items()}
        data[k] = [x[0] for x in sorted(差v.items(), key=lambda x: x[1], reverse=True)[:3]]
    df = pd.DataFrame(data, index=['top1', 'top2', 'top3'])
    output.write('# 模型偏好标签: \n' + df.to_markdown() + '\n\n')
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
        for 种类, d in [('胸部', breasts), ('头发', hair)]:
            总个数 = 0
            总分数 = 0
            for h, hv in d.items():
                总个数 += v.get(h, 0)
                总分数 += v.get(h, 0) * hv
            q.setdefault(k, {}).setdefault(种类, 总分数/总个数)
        q[k]['颜色'] = max([(v.get(x, 0), x) for x in 头发颜色])[1].removesuffix('_hair').replace('blonde', 'yellow')
    from bokeh.plotting import figure, show
    from bokeh.models.annotations import Label
    模型 = [*模型标签计数]
    x = [q[i]['胸部'] for i in 模型]
    y = [q[i]['头发'] for i in 模型]
    x, y = _分离(x, y, t=0.004)
    color = [q[i]['颜色'] for i in 模型]
    p = figure(title="散点图", x_axis_label="胸部大小", y_axis_label="头发长度", x_range = (min(x)-0.005, max(x)+0.01), width=1024, height=512)
    p.circle(x, y, size=10, color=color)
    for i in range(len(x)):
        label = Label(x=x[i]+0.0016, y=y[i]-0.0058, text=模型[i], text_font_size='9pt')
        p.add_layout(label)
    show(p)



def 导出多标签():
    l = orjson.loads(open('savedata/记录_多标签.json', encoding='utf-8').read())
    m = {}
    for d in l:
        n = len(d['标签组'])
        model = _模型改名(d['参数']['override_settings']['sd_model_checkpoint'])
        m.setdefault((model, n), {'相似度': [], '分数': []})
        m[model, n]['相似度'].extend(d['相似度'])
        m[model, n]['分数'].extend(d['分数'])
    all_model, all_n = zip(*m.keys())
    all_model = sorted({*all_model}, key=lambda x: x if 'XL' in x else '0' + x)
    all_n = sorted({*all_n})

    data = {}
    data2 = {}
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
                acc = (a > 0.001).sum() / len(a.flatten())
                data[model].append(round(acc, 3))
                data2[model].append(round(1 - np.array(m[model, n]['相似度']).mean(), 3))
                
    # data = {k: v for k, v in data.items() if k in readme要的}
    # data2 = {k: v for k, v in data2.items() if k in readme要的}

    output.write('# 模型对标签个数-准确率: \n' + pd.DataFrame(_加粗(data, all_n), index=all_n).to_markdown() + '\n\n')
    output.write('# 模型对标签个数-多样性: \n' + pd.DataFrame(_加粗(data2, all_n), index=all_n).to_markdown() + '\n\n')

    # from bokeh.plotting import figure, show
    # from bokeh.models.annotations import Label
    # x = [data[i][4] for i in all_model]
    # y = [data2[i][4] for i in all_model]
    # x, y = _分离(x, y)
    # p = figure(title="散点图", x_axis_label="准确度", y_axis_label="多样性", x_range = (min(x)-0.005, max(x)+0.01), width=1024, height=512)
    # p.circle(x, y, size=10, color="blue", alpha=0.5)
    # for i in range(len(x)):
    #     label = Label(x=x[i]+0.001, y=y[i]-0.0013, text=all_model[i], text_font_size='8pt')
    #     p.add_layout(label)
    # show(p)



def 导出不同参数():
    l = orjson.loads(open('savedata/记录_不同参数.json', encoding='utf-8').read())
    d = {}
    for i in l:
        model = _模型改名(i['参数']['override_settings']['sd_model_checkpoint'])
        参数 = i['参数']
        x = f'{参数["width"]}×{参数["height"]}'
        y = 参数['steps']
        d.setdefault((model, x, y), []).extend(i['分数'])
    dd = {}
    for (model, x, y), 分数 in d.items(): 
        a = np.array(分数)
        acc = (a > 0.001).sum() / len(a.flatten())
        dd.setdefault(model, {})[x, y] = acc
    
    output.write('# 不同模型在不同尺寸和step下的准确率: \n\n')
    for model in dd:
        all_x, all_y = map(sorted, map(set, zip(*dd[model].keys())))
        data = {}
        for x in all_x:
            data[x] = []
            for y in all_y:
                data[x].append(dd[model][x, y])
        output.write(f'## {model}\n')
        output.write(f'{pd.DataFrame(data, index=all_y).to_markdown()}\n\n')


导出单标签()
导出单标签2()
导出多标签()
导出不同参数()
