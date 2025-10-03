import yaml
from typing import Optional, MutableMapping
from dataclasses import dataclass


def _sort_key(p):
    return {
        'sd': 0,
        'sdxl': 1,
    }.get(p[1].类型, 999), p[1].名字.lower()


@dataclass
class 模型:
    名字: str
    类型: str
    vae: Optional[str] = None
    隐藏: bool = False
    额外参数: Optional[dict] = None
    动漫: bool = True


模型池 = {}
for k, v in yaml.safe_load(open('模型.yaml', encoding='utf8')).items():
    m = 模型(名字=k, **v)
    if m.隐藏:
        continue
    模型池[k] = m
模型池 = {k: v for k, v in sorted(模型池.items(), key=_sort_key)}
