import json
import copy
import numpy as np

import pandas as pd


output = open('好.md', 'w', encoding='utf8')


def _模型改名(x):    # 为了让表格在 GitHub 上显示更好看
    return {
        'AnythingV5Ink_ink': 'A5Ink',
        'anything-v4.5-pruned-fp32': 'A4.5',
        'Counterfeit-V2.2': 'CF2.2',
        'Counterfeit-V3.0_fp16': 'CF3.0',
        'cosplaymix_v20': 'CM20',
        'novelailatest-pruned': 'novelai',
        'sweetfruit_melon.safetensors_v1.0': 'SF1.0',
        'bluePencil_v10': 'BP10',
        'blue_pencil-XL-v0.3.1': 'BPXL0.3.1',
        'CounterfeitXL-V1.0': 'CFXL1.0',
        'cuteyukimixAdorable_midchapter3': 'CYM3',
        'Counterfeit-V2.5_pruned': 'CF2.5',
        'perfectWorld_v2Baked': 'PW2',
        'perfectWorld_v6Baked': 'PW6',
        'meinamix_meinaV11': 'MM11',
        'cetusMix_Whalefall2': 'CMWF2',
    }.get(x, x)


def _加粗(data: dict[str, list], yy):
    data = copy.deepcopy(data)
    xx = [*data.keys()]
    for i, _ in enumerate(yy):
        top = sorted([data[x][i] for x in xx], reverse=True, key=lambda x: x if isinstance(x, float|int) else -1)[:3]
        for x in xx:
            if data[x][i] in top and data[x][i] != '-':
                data[x][i] = f'**{data[x][i]}**'
    return data


def 导出单标签():
    l = json.load(open('savedata/记录.json', encoding='utf-8'))
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

    目录 = json.load(open('data/目录.json', encoding='utf-8'))
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
    df = pd.DataFrame(_加粗(data, sorted_目录), index=[目录[i]['name'] for i in sorted_目录])
    output.write('# 模型对标签类别-准确率: \n' + df.to_markdown() + '\n\n')


def 导出多标签():
    l = json.load(open('savedata/记录_多标签.json', encoding='utf-8'))
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
    output.write('# 模型对标签个数-准确率: \n' + pd.DataFrame(_加粗(data, all_n), index=all_n).to_markdown() + '\n\n')
    output.write('# 模型对标签个数-多样性: \n' + pd.DataFrame(_加粗(data2, all_n), index=all_n).to_markdown() + '\n\n')


导出单标签()
导出多标签()
