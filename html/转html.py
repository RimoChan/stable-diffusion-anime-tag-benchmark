import markdown
from pathlib import Path

此处 = Path(__file__) / '..'


for i in [此处 / '../测试结果/模型对不同系列的准确率.md', 此处 / '../测试结果/模型对不同角色的准确率.md', 此处 / '../测试结果/模型对标签类别-准确率768.md', 此处 / '../测试结果/模型对标签大类-准确率768.md']:
    md_text = open(i, 'r', encoding='utf8').read()
    html = markdown.markdown(md_text, extensions=['tables'])
    with open(此处 / (Path(i).stem + '.html'), 'w', encoding='utf8') as f:
        f.write('<style>table{table-layout: fixed;border-collapse: collapse;}td, th{border: 1px solid #777;padding: 3px 7px;}</style>'+html)
