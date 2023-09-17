import json

import pandas as pd


output = open('1.md', 'w', encoding='utf8')

l = json.load(open('记录.json', encoding='utf-8'))
m = {}
all_model = set()
all_tag = set()
for d in l:
    model = d['参数']['override_settings']['sd_model_checkpoint']
    好 = len([i for i in d['分数'] if i > 0.1])
    n = len(d['分数'])
    assert (model, d['标签']) not in m
    m[model, d['标签']] = 好, n
    all_model.add(model)
    all_tag.add(d['标签'])
all_model = sorted(all_model)
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
output.write(df.to_markdown() + '\n\n')


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
            if n <= 16:     # 不置信
                data[model].append('-')
            else:
                data[model].append(round(好 / n, 3))
for i, tag in enumerate(sorted_目录):
    top = sorted([data[model][i] for model in all_model], reverse=True)[:1]
    for model in all_model:
        if data[model][i] in top and data[model][i] != '-':
            data[model][i] = f'**{data[model][i]}**'
df = pd.DataFrame(data, index=[目录[i]['name'] for i in sorted_目录])
output.write(df.to_markdown())
