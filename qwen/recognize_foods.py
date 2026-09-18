'''
用 Qwen3-VL 识别图片中的食品（类似 recognize-anything / RAM 的打标签能力）。

独立脚本，暂不接入 food_pipeline。输入一张图片，输出食品标签列表。

用法:
  1. 把 API Key 写到本目录的 .env（见 .env.example）
  2. pip install -r requirements.txt
  3. python recognize_foods.py --image path/to/photo.jpg

密钥读取优先级（高 -> 低）:
  --api-key 参数 > 环境变量 DASHSCOPE_API_KEY > qwen/.env
'''
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / '.env')

# 并行科技 Paratera 上的模型 ID（大小写敏感）；阿里云百炼则为小写
# qwen3-vl-235b-a22b-instruct
DEFAULT_MODEL = 'Qwen3-VL-235B-A22B-Instruct'
DEFAULT_BASE_URL = 'https://llmapi.paratera.com/v1'

_SYSTEM_PROMPT = '''\
你是食品图像识别助手，输出要直接喂给目标检测模型（GroundingDINO）做框选。
只输出 JSON，不要 markdown 代码块，不要解释。格式严格如下：
{
  "all_tags": ["图中主要可见物体的英文标签，含非食品"],
  "items": [
    {
      "label": "检测用英文类别名",
      "label_zh": "中文名",
      "count": 1,
      "separable": true
    }
  ]
}

命名与计数规则（很重要，必须严格遵守）：
1) 可明显拆分成独立个体的食物（荔枝、香蕉、苹果、草莓、鸡蛋、整只虾等）：
   - separable=true
   - label 必须用 "single" + 单数名词，例如：
       "single lychee" / "single banana" / "single apple" / "single strawberry"
   - 禁止使用不定冠词 "a" / "an"（例如不要写 "a lychee"）
   - 禁止复数（不要写 "lychees" / "bananas"）
   - 禁止只用光秃秃的名词（不要写 "lychee"）；必须带 single，强调逐个框选
   - count = 能数清的个体数量（按个/根/只计）
2) 混在一起、堆叠、或难以逐个框出的食物（虾仁堆、沙拉、炒菜、一盘花生等）：
   - separable=false
   - label 用简短类别名，可加少量修饰，如 "shrimp" / "peeled shrimp" / "fried rice"
   - 不要加 "single" / "a" / "an"
   - count 通常为 1（整堆/整盘作为一个目标）；只有当有多堆明显分开的同类时才 >1
3) 没有食品时 items 为空数组。
4) 不要输出餐具/人（table/plate/bowl/person）。
5) 同类只出现一条；count 表示该类要检测的目标数。
'''

_SPOILAGE_LEVELS = ('fresh', 'spoiling', 'spoiled')
_KNOWN_GASES = {
    'ethylene': 'Ethylene',
    'ethanol': 'Ethanol',
    'ammonia': 'Ammonia',
    'hydrogen sulfide': 'Hydrogen Sulfide',
    'h2s': 'Hydrogen Sulfide',
    'methanethiol': 'Methanethiol',
    'ch3sh': 'Methanethiol',
    'voc': 'VOC',
}

_SPOILAGE_SYSTEM_PROMPT = '''\
你是食品新鲜度 / 可食用性评估助手。输入是裁剪后的单个（或一小堆）食物图像。
只输出 JSON，不要 markdown 代码块，不要解释。格式严格如下：
{
  "spoilageScore": 0.0,
  "spoilageLevel": "fresh",
  "message": "面向 App 的一句中文状态说明。",
  "producedGases": ["Ethylene"]
}

spoilageScore（浮点数，必须在 0~1）：
- 0 = 非常新鲜、完全可安全食用
- 越接近 1 = 越不宜食用
- 1 = 已不能食用（重要：1 表示「不可食用」的阈值，不是「完全腐烂」；
  一旦腐败到不能安全食用的程度，分数保持为 1，即使外观继续恶化也不再升高）
- 在新鲜与不可食用之间按严重程度连续打分

spoilageLevel 只能是以下之一：
- "fresh"：新鲜可食
- "spoiling"：开始转差，建议尽快食用
- "spoiled"：已不宜食用（对应高 spoilageScore，接近或等于 1）

message：一句简短中文说明，面向 App 展示（可提及食物名与大致状态）。

producedGases：该食物在此阶段通常相关的气体英文名列表，优先从下列选取：
Ethylene, Ethanol, Ammonia, Hydrogen Sulfide, Methanethiol, VOC
新鲜时可给 ["Ethylene"]；明显腐败时按品类选择合理产气组合。
'''


def _resolve_api_key(cli_key: str | None) -> str:
    key = (cli_key or os.getenv('DASHSCOPE_API_KEY') or '').strip()
    if not key:
        raise ValueError(
            '未找到 API Key。请任选其一：'
            ' 1) 在 qwen/.env 写入 DASHSCOPE_API_KEY=sk-...'
            ' 2) 设置环境变量 DASHSCOPE_API_KEY'
            ' 3) 传参 --api-key sk-...'
        )
    return key


def _image_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f'图片不存在: {image_path}')

    mime, _ = mimetypes.guess_type(path.name)
    if mime not in {'image/jpeg', 'image/png', 'image/webp', 'image/gif',
                    'image/bmp'}:
        # 多数相机输出 jpg；猜不到时按 jpeg
        mime = 'image/jpeg'

    b64 = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:{mime};base64,{b64}'


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 模型偶发包一层 ```json ... ```
    fence = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if fence:
        return json.loads(fence.group(1).strip())

    start, end = text.find('{'), text.rfind('}')
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f'无法从模型回复中解析 JSON:\n{text}')


def _normalize_item_label(label: str, separable: bool | None = None) -> str:
    """Normalize detection label; map a/an X -> single X for individuals."""
    s = ' '.join(str(label).strip().split())
    if not s:
        return ''
    s = s.lower()

    # Strip leading articles / single, recover the noun phrase.
    noun = s
    for prefix in ('a ', 'an ', 'the ', 'single '):
        if noun.startswith(prefix):
            noun = noun[len(prefix):].strip()
            break

    # Explicit separable flag from model, or heuristic: had a/an/single.
    looks_individual = separable is True or any(
        s.startswith(p) for p in ('a ', 'an ', 'the ', 'single ')
    )
    if separable is False:
        return noun  # mixed pile: bare / modified category name

    if looks_individual or separable is True:
        # Force singular-ish: drop a trailing simple plural 's' on last word
        # only when it's a single-token food (lychees -> lychee). Keep words
        # ending in 'ss' / 'us' / 'is' untouched.
        parts = noun.split()
        if len(parts) == 1:
            w = parts[0]
            if (w.endswith('s') and not w.endswith(('ss', 'us', 'is'))
                    and len(w) > 3):
                # bananas -> banana, lychees -> lychee; fruits -> fruit
                if w.endswith('es') and w[:-2].endswith(('ch', 'sh', 'x', 'z')):
                    w = w[:-2]
                elif w.endswith('ies') and len(w) > 4:
                    w = w[:-3] + 'y'
                else:
                    w = w[:-1]
                parts = [w]
            noun = parts[0]
        return f'single {noun}'.strip()

    return noun


def _parse_items(parsed: dict) -> tuple[list[str], list[str], dict[str, int]]:
    """Return (foods, foods_zh, food_counts) from model JSON."""
    foods, foods_zh, food_counts = [], [], {}

    items = parsed.get('items')
    if isinstance(items, list) and items:
        for it in items:
            if not isinstance(it, dict):
                continue
            label = _normalize_item_label(
                it.get('label', ''),
                separable=it.get('separable'),
            )
            if not label:
                continue
            zh = str(it.get('label_zh') or it.get('zh') or '').strip()
            try:
                count = int(it.get('count', 1))
            except (TypeError, ValueError):
                count = 1
            count = max(count, 1)
            # If model forgot separable=true but count>1 with bare noun,
            # treat as individuals.
            if (it.get('separable') is None and count > 1
                    and not label.startswith('single ')):
                label = _normalize_item_label(label, separable=True)
            if label in food_counts:
                # Same label twice: take the larger count, keep first zh.
                food_counts[label] = max(food_counts[label], count)
                continue
            foods.append(label)
            foods_zh.append(zh)
            food_counts[label] = count
        return foods, foods_zh, food_counts

    # Backward compatible: old {foods, foods_zh} without counts.
    foods = [_normalize_item_label(t) for t in parsed.get('foods', [])]
    foods = [t for t in foods if t]
    foods_zh = [str(t).strip() for t in parsed.get('foods_zh', [])]
    seen, uniq, uniq_zh = set(), [], []
    for i, f in enumerate(foods):
        if f in seen:
            continue
        seen.add(f)
        uniq.append(f)
        uniq_zh.append(foods_zh[i] if i < len(foods_zh) else '')
        food_counts[f] = 1
    return uniq, uniq_zh, food_counts


def _client_and_model(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> tuple[OpenAI, str]:
    key = _resolve_api_key(api_key)
    url = (base_url or os.getenv('DASHSCOPE_BASE_URL') or DEFAULT_BASE_URL).rstrip('/')
    if not url.endswith('/v1'):
        url = url + '/v1'
    model_name = model or os.getenv('QWEN_VL_MODEL') or DEFAULT_MODEL
    return OpenAI(api_key=key, base_url=url), model_name


def _normalize_spoilage_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    if score != score:  # NaN
        score = 0.0
    return round(min(max(score, 0.0), 1.0), 4)


def _level_from_score(score: float) -> str:
    if score >= 0.7:
        return 'spoiled'
    if score >= 0.35:
        return 'spoiling'
    return 'fresh'


def _normalize_produced_gases(gases) -> list[str]:
    if not isinstance(gases, list):
        return ['Ethylene']
    out, seen = [], set()
    for g in gases:
        key = str(g or '').strip().lower()
        if not key:
            continue
        name = _KNOWN_GASES.get(key) or str(g).strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out or ['Ethylene']


def _normalize_spoilage_result(parsed: dict, food_label: str = '') -> dict:
    score = _normalize_spoilage_score(parsed.get('spoilageScore'))
    level = str(parsed.get('spoilageLevel') or '').strip().lower()
    if level not in _SPOILAGE_LEVELS:
        level = _level_from_score(score)
    # Keep level/score roughly consistent at the inedible end.
    if score >= 0.95 and level == 'fresh':
        level = 'spoiled'
    elif score <= 0.15 and level == 'spoiled':
        level = 'fresh'

    message = str(parsed.get('message') or '').strip()
    if not message:
        label = (food_label or '食物').strip() or '食物'
        level_zh = {'fresh': '新鲜', 'spoiling': '开始转差',
                    'spoiled': '已不宜食用'}.get(level, level)
        message = f'{label}看起来{level_zh}（评分 {score:.2f}）。'

    return {
        'spoilageScore': score,
        'spoilageLevel': level,
        'message': message,
        'producedGases': _normalize_produced_gases(parsed.get('producedGases')),
        'success': True,
    }


def assess_spoilage(
    image_path: str,
    food_label: str = '',
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.1,
) -> dict:
    """用 VL 评估单张裁剪图的新鲜度 / 可食用性。

    返回:
      spoilageScore (0~1，越大越不宜食用；1=不可食用阈值，非「完全腐烂」),
      spoilageLevel ('fresh'|'spoiling'|'spoiled'),
      message, producedGases, raw, model, image, success
    """
    client, model_name = _client_and_model(api_key, model, base_url)
    data_url = _image_data_url(image_path)
    label = (food_label or '').strip() or 'unknown food'
    user_text = (
        f'The crop shows: {label}. '
        'Assess edible quality / spoilage. '
        'spoilageScore: 0=very fresh, approaching 1=less edible, '
        '1=inedible (not "fully rotten"; once unsafe to eat, stay at 1). '
        'Also give spoilageLevel, a short Chinese message, producedGases. '
        'Output JSON only.'
    )

    completion = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {'role': 'system', 'content': _SPOILAGE_SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': data_url}},
                    {'type': 'text', 'text': user_text},
                ],
            },
        ],
    )

    raw = completion.choices[0].message.content or ''
    parsed = _extract_json(raw)
    result = _normalize_spoilage_result(parsed, food_label=label)
    result.update({
        'raw': raw,
        'model': model_name,
        'image': os.path.abspath(image_path),
    })
    return result


def recognize_foods(
    image_path: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.1,
) -> dict:
    """调用视觉模型，返回 {all_tags, foods, foods_zh, food_counts, raw}。"""
    client, model_name = _client_and_model(api_key, model, base_url)
    data_url = _image_data_url(image_path)

    completion = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': data_url}},
                    {
                        'type': 'text',
                        'text': (
                            '请识别这张图片里的食品。'
                            '可拆分的个体必须用 "single lychee" / "single banana" '
                            '这类标签（禁止 a/an），并给出 count；'
                            '混在一起的用类别名且不要加 single，count 一般为 1。'
                            '严格按约定 JSON 输出。'
                        ),
                    },
                ],
            },
        ],
    )

    raw = completion.choices[0].message.content or ''
    parsed = _extract_json(raw)

    all_tags = [str(t).strip() for t in parsed.get('all_tags', []) if str(t).strip()]
    foods, foods_zh, food_counts = _parse_items(parsed)

    # 去重 all_tags，保序
    seen, uniq_tags = set(), []
    for x in all_tags:
        if x not in seen:
            seen.add(x)
            uniq_tags.append(x)

    return {
        'all_tags': uniq_tags,
        'foods': foods,
        'foods_zh': foods_zh,
        'food_counts': food_counts,
        'raw': raw,
        'model': model_name,
        'image': os.path.abspath(image_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Qwen3-VL 食品识别（类似 RAM 打标签）')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--api-key', default=None,
                        help='百炼 API Key（默认读 .env / 环境变量）')
    parser.add_argument('--model', default=None,
                        help='模型名（默认读 .env 的 QWEN_VL_MODEL，'
                             f'再退回 {DEFAULT_MODEL}）')
    parser.add_argument('--base-url', default=None,
                        help='API base URL（默认国内 DashScope 兼容模式）')
    parser.add_argument('--output', default=None,
                        help='把结果写入 JSON 文件；默认打印到终端')
    parser.add_argument('--temperature', type=float, default=0.1)
    args = parser.parse_args()

    result = recognize_foods(
        args.image,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )

    # 终端摘要（对齐 food_pipeline 的打印风格）
    print(f'model    : {result["model"]}')
    print(f'image    : {result["image"]}')
    print('all tags :', ' | '.join(result['all_tags']) or '(none)')
    print('foods    :', ' | '.join(result['foods']) or '(none)')
    if result.get('food_counts'):
        counts = ', '.join(f'{k}×{v}' for k, v in result['food_counts'].items())
        print('counts   :', counts)
    if result['foods_zh']:
        print('foods_zh :', ' | '.join(result['foods_zh']))

    payload = {k: v for k, v in result.items() if k != 'raw'}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding='utf-8')
        print(f'\nWrote {out}')
    else:
        print('\n' + text)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
