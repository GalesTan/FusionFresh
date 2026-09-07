'''
Food pipeline: VL API / RAM++ -> food tags -> GroundingDINO.

Given any photo, this script:
  1. tags foods via a vision LLM API when available, otherwise RAM++,
  2. feeds the food tags to GroundingDINO as its text prompt,
  3. detects + counts each food and crops every detection.

Example:
in powershell:
  python food_pipeline.py --image demo/demo5.jpg --gdino-checkpoint vendor/GroundingDINO/weights/groundingdino_swint_ogc.pth --output-dir outputs/demo5 --cpu-only
'''
import argparse
import json
import os
import random
import sys

# GroundingDINO 需要从 HuggingFace 拉 bert-base-uncased；国内直连常超时。
# 未手动设置时默认走镜像；可在启动前自行 export HF_ENDPOINT 覆盖。
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import numpy as np
import torch
from PIL import Image

# --- make sibling modules importable ---------------------------------------
# Layout (this file stays at project root so the pipeline API is unchanged):
#   vendor/recognize-anything  -> ram
#   vendor/GroundingDINO       -> groundingdino
#   spoilage/lychee            -> lychee.predictor
#   spoilage/shrimp            -> predictor / gas_fusion
#   qwen/                      -> qwen.recognize_foods
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, 'vendor')
sys.path.insert(0, os.path.join(_VENDOR, 'recognize-anything'))
sys.path.insert(0, os.path.join(_VENDOR, 'GroundingDINO'))
sys.path.insert(0, os.path.join(_HERE, 'spoilage', 'lychee'))
sys.path.insert(0, os.path.join(_HERE, 'spoilage', 'shrimp'))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from groundingdino.util.inference import load_model, predict  # noqa: E402
import groundingdino.datasets.transforms as T  # noqa: E402

try:
    from qwen.recognize_foods import (
        recognize_foods as vl_recognize_foods,
        assess_spoilage as vl_assess_spoilage,
    )
except ImportError:  # openai / dotenv missing
    vl_recognize_foods = None
    vl_assess_spoilage = None

# RAM++ is optional at import time so `--tagger vl` still works if the local
# transformers stack is broken; fallback / `--tagger ram` load it lazily.
_ram_imports = None


def _ensure_ram():
    global _ram_imports
    if _ram_imports is not None:
        return _ram_imports
    from ram import get_transform, inference_ram, filter_food_tags
    from ram.models import ram_plus
    _ram_imports = {
        'get_transform': get_transform,
        'inference_ram': inference_ram,
        'filter_food_tags': filter_food_tags,
        'ram_plus': ram_plus,
    }
    return _ram_imports

_DEFAULT_GDINO_CFG = os.path.join(
    _VENDOR, 'GroundingDINO', 'groundingdino', 'config',
    'GroundingDINO_SwinT_OGC.py')

_RAM_DATA = os.path.join(_VENDOR, 'recognize-anything', 'ram', 'data')


def _load_en2cn():
    """Map English RAM tags -> Chinese, using the line-aligned tag lists."""
    en_path = os.path.join(_RAM_DATA, 'ram_tag_list.txt')
    cn_path = os.path.join(_RAM_DATA, 'ram_tag_list_chinese.txt')
    mapping = {}
    with open(en_path, encoding='utf-8') as fe, open(cn_path, encoding='utf-8') as fc:
        for en, cn in zip(fe, fc):
            en = en.strip().lower()
            # Chinese entries may look like "橙子/橙色 " -> take first, strip.
            cn = cn.strip().split('/')[0].strip()
            if en and cn:
                mapping[en] = cn
    return mapping


_EN2CN = _load_en2cn()


def to_chinese(label, overrides=None):
    """Chinese label for an English food tag; falls back to the English tag."""
    key = label.lower()
    if overrides and key in overrides:
        return overrides[key]
    return _EN2CN.get(key, label)


# --- Spoilage scoring -------------------------------------------------------
# Lychee / shrimp: dedicated models.
# Other foods: Qwen VL on each crop (score / level / message / gases).
# Shrimp-only scenes may fuse VOC/C2H5OH/H2S/NH3 history (see gas_fusion).
#
# producedGases：按食物种类固定关联气体（与腐败等级无关）。
_FOOD_RELATED_GASES = {
    'shrimp': ['VOC', 'Ethanol', 'Hydrogen Sulfide', 'Ammonia'],
    'lychee': ['VOC', 'Ethanol'],
    'banana': ['VOC', 'Ethanol', 'Ethylene'],
}
_DEFAULT_RELATED_GASES = ['VOC', 'Ethanol']

_SPOILAGE = {
    'fresh': {
        'weight': 70,
        'messages': ['Perfectly ripe.', 'Fresh and ripe.',
                     'In great condition.', 'Freshly stocked.'],
    },
    'spoiling': {
        'weight': 22,
        'messages': ['Past peak — use soon.', 'Starting to soften.',
                     'Early signs of spoilage.'],
    },
    'spoiled': {
        'weight': 8,
        'messages': ['Spoiled — discard.', 'No longer safe to eat.',
                     'Significant spoilage detected.'],
    },
}
_LEVELS = list(_SPOILAGE)
_WEIGHTS = [_SPOILAGE[lvl]['weight'] for lvl in _LEVELS]

# LycheePredictor rating_id → app spoilageLevel
_LYCHEE_RATING_TO_LEVEL = {0: 'fresh', 1: 'spoiling', 2: 'spoiled'}
_LYCHEE_LEVEL_MESSAGES = {
    'fresh': 'Lychee early stage - fresh (score {score:.2f}).',
    'spoiling': 'Lychee mid stage - use soon (score {score:.2f}).',
    'spoiled': 'Lychee late stage - spoiled (score {score:.2f}).',
}

# shrimp_infer native level → app spoilageLevel (uncertain ≈ mid / spoiling)
_SHRIMP_LEVEL_TO_APP = {
    'fresh': 'fresh',
    'uncertain': 'spoiling',
    'spoiled': 'spoiled',
}
_SHRIMP_LEVEL_MESSAGES = {
    'fresh': '虾仁新鲜，可放心食用。',
    'spoiling': '虾仁处于中期，建议尽快食用。',
    'spoiled': '虾仁已腐败，不宜食用。',
}

_lychee_predictor = None
_shrimp_predictor = None


def _related_gases_for_label(label):
    """Return the fixed related-gas list for a food label."""
    if _is_shrimp(label):
        return list(_FOOD_RELATED_GASES['shrimp'])
    if _is_lychee(label):
        return list(_FOOD_RELATED_GASES['lychee'])
    if _is_banana(label):
        return list(_FOOD_RELATED_GASES['banana'])
    return list(_DEFAULT_RELATED_GASES)


def _random_spoilage(label=None):
    """Return (spoilageLevel, message, producedGases, spoilageScore).

    spoilageScore is None for placeholders (no real model).
    """
    level = random.choices(_LEVELS, weights=_WEIGHTS, k=1)[0]
    cfg = _SPOILAGE[level]
    return (level, random.choice(cfg['messages']),
            _related_gases_for_label(label), None)


def _is_lychee(label):
    """True if the detection label refers to lychee / 荔枝."""
    text = (label or '').strip().lower()
    if not text:
        return False
    if '荔枝' in (label or ''):
        return True
    core = _strip_article(text)
    return any(tok in text or tok in core for tok in ('lychee', 'litchi'))


def _is_shrimp(label):
    """True if the detection label refers to shrimp / 虾仁."""
    raw = label or ''
    text = raw.strip().lower()
    if not text:
        return False
    if any(tok in raw for tok in ('虾仁', '虾米', '明虾', '基围虾')):
        return True
    # bare "虾" but not other compounds we don't handle
    if '虾' in raw and '龙虾' not in raw:
        return True
    core = _strip_article(text)
    return any(tok in text or tok in core
               for tok in ('shrimp', 'prawn', 'peeled shrimp'))


def _is_banana(label):
    """True if the detection label refers to banana / 香蕉."""
    raw = label or ''
    text = raw.strip().lower()
    if not text:
        return False
    if '香蕉' in raw:
        return True
    core = _strip_article(text)
    return any(tok in text or tok in core for tok in ('banana', 'bananas'))


def get_lychee_predictor(device=None):
    """Lazy-load a shared LycheePredictor (DINOv2 + PCA + Ridge)."""
    global _lychee_predictor
    from lychee.predictor import LycheePredictor
    want = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if (_lychee_predictor is None
            or str(_lychee_predictor.device) != str(want)):
        _lychee_predictor = LycheePredictor(device=want)
    return _lychee_predictor


def get_shrimp_predictor(device=None, score_mode='anchor'):
    """Lazy-load a shared ShrimpPredictor (DINOv2 + PCA + anchor/Ridge)."""
    global _shrimp_predictor
    from predictor import ShrimpPredictor
    want = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if (_shrimp_predictor is None
            or str(_shrimp_predictor.device) != str(want)
            or _shrimp_predictor.score_mode != score_mode):
        _shrimp_predictor = ShrimpPredictor(device=want, score_mode=score_mode)
    return _shrimp_predictor


def _spoilage_from_lychee_score(pred):
    """Map LycheePredictor → (spoilageLevel, message, producedGases, score)."""
    rating_id = int(pred.get('rating_id', 1))
    level = _LYCHEE_RATING_TO_LEVEL.get(rating_id, 'spoiling')
    score = round(float(pred.get('score', 0.0)), 4)
    message = _LYCHEE_LEVEL_MESSAGES[level].format(score=score)
    return level, message, list(_FOOD_RELATED_GASES['lychee']), score


def _spoilage_from_shrimp_score(pred):
    """Map ShrimpPredictor → (spoilageLevel, message, producedGases, score)."""
    native = (pred.get('spoilage_level') or 'uncertain').lower()
    level = _SHRIMP_LEVEL_TO_APP.get(native, 'spoiling')
    score = round(float(pred.get('score', 0.0)), 4)
    message = _SHRIMP_LEVEL_MESSAGES[level]
    return level, message, list(_FOOD_RELATED_GASES['shrimp']), score


def _spoilage_from_vl(pred, label=None):
    """Map VL assess_spoilage → (spoilageLevel, message, producedGases, score)."""
    level = str(pred.get('spoilageLevel') or 'spoiling').lower()
    if level not in _SPOILAGE:
        level = 'spoiling'
    score = round(float(pred.get('spoilageScore', 0.0)), 4)
    score = min(max(score, 0.0), 1.0)
    message = str(pred.get('message') or '').strip() or (
        f'VL spoilage {level} (score {score:.2f}).')
    # 已知食物用固定关联气体；未知食物才回退 VL 返回值
    if _is_shrimp(label) or _is_lychee(label) or _is_banana(label):
        gases = _related_gases_for_label(label)
    else:
        gases = pred.get('producedGases')
        if not isinstance(gases, list) or not gases:
            gases = list(_DEFAULT_RELATED_GASES)
        else:
            gases = [str(g).strip() for g in gases if str(g).strip()]
    return level, message, gases, score


def _resolve_crop_path(det, output_dir):
    """Absolute path to a cropped detection, or None if unavailable."""
    crop = det.get('crop')
    if not crop:
        return None
    if os.path.isabs(crop) and os.path.isfile(crop):
        return crop
    if output_dir:
        candidate = os.path.join(output_dir, crop)
        if os.path.isfile(candidate):
            return candidate
    if os.path.isfile(crop):
        return crop
    return None


def _spoilage_for_detection(det, output_dir=None, lychee_predictor=None,
                            shrimp_predictor=None, device=None):
    """Score one detection: DINOv2 for lychee/shrimp, VL for other foods.

    Returns (spoilageLevel, message, producedGases, spoilageScore).
    Falls back to random placeholder if the preferred path fails.
    """
    label = det.get('label')
    crop_path = _resolve_crop_path(det, output_dir)

    if _is_lychee(label):
        if crop_path is None:
            print(f'[lychee] no crop for {label}; using placeholder')
            return _random_spoilage(label)
        try:
            predictor = lychee_predictor or get_lychee_predictor(device=device)
            pred = predictor.predict(crop_path, calibrate=False,
                                     skip_standardize=False)
            if not pred.get('success', True):
                raise RuntimeError(pred.get('error') or 'predict failed')
            return _spoilage_from_lychee_score(pred)
        except Exception as exc:
            print(f'[lychee] score failed on {crop_path}: {exc}; '
                  'using placeholder')
            return _random_spoilage(label)

    if _is_shrimp(label):
        if crop_path is None:
            print(f'[shrimp] no crop for {label}; using placeholder')
            return _random_spoilage(label)
        try:
            predictor = shrimp_predictor or get_shrimp_predictor(device=device)
            pred = predictor.predict(crop_path)
            if not pred.get('success', True):
                raise RuntimeError(pred.get('error') or 'predict failed')
            return _spoilage_from_shrimp_score(pred)
        except Exception as exc:
            print(f'[shrimp] score failed on {crop_path}: {exc}; '
                  'using placeholder')
            return _random_spoilage(label)

    # Other foods: VL spoilage assessment on the crop.
    if crop_path is None:
        print(f'[vl-spoilage] no crop for {label}; using placeholder')
        return _random_spoilage(label)
    if vl_assess_spoilage is None:
        print(f'[vl-spoilage] VL unavailable for {label}; using placeholder')
        return _random_spoilage(label)
    try:
        pred = vl_assess_spoilage(crop_path, food_label=label or '')
        if not pred.get('success', True):
            raise RuntimeError(pred.get('error') or 'assess_spoilage failed')
        return _spoilage_from_vl(pred, label=label)
    except Exception as exc:
        print(f'[vl-spoilage] failed on {crop_path}: {exc}; '
              'using placeholder')
        return _random_spoilage(label)


def build_detection_results(image_path, detections, label_zh=None,
                            output_dir=None, lychee_predictor=None,
                            shrimp_predictor=None, device=None,
                            gas_data_root=None, apply_shrimp_gas=True):
    """Format pipeline detections into the app's detection_results schema.

    Real fields: source_image, label, type_probability, boundingBox,
    spoilageScore (0-1; 1 = inedible threshold, not fully rotten).
    Lychee/shrimp use dedicated models; other foods use VL on each crop
    (spoilageLevel / message / producedGases included).

    When every detection is shrimp and apply_shrimp_gas is True, fuse
    esp32 gas history (VOC/C2H5OH → mid, H2S/NH3 → spoiled).
    """
    source_image = os.path.basename(image_path)
    detected_foods = []
    any_spoiled = False
    for det in detections:
        level, message, gases, score = _spoilage_for_detection(
            det, output_dir=output_dir, lychee_predictor=lychee_predictor,
            shrimp_predictor=shrimp_predictor, device=device)
        any_spoiled = any_spoiled or level == 'spoiled'
        detected_foods.append({
            'source_image': source_image,
            'label': to_chinese(det['label'], overrides=label_zh),
            'type_probability': det['probability'],
            'spoilageScore': score,
            'spoilageLevel': level,
            'boundingBox': det['boundingBox'],
            'message': message,
            'producedGases': gases,
        })

    shrimp_gas_meta = None
    only_shrimp = bool(detections) and all(
        _is_shrimp(d.get('label')) for d in detections)
    if apply_shrimp_gas and only_shrimp and detected_foods:
        try:
            from gas_fusion import (
                DEFAULT_THRESHOLDS,
                apply_gas_override_to_food,
                evaluate_shrimp_gas_rules,
            )
            data_root = gas_data_root or os.path.join(
                _HERE, 'collect', 'data')
            override = evaluate_shrimp_gas_rules(data_root)
            if override is not None:
                thresholds = dict(DEFAULT_THRESHOLDS)
                try:
                    predictor = shrimp_predictor or get_shrimp_predictor(
                        device=device)
                    thr = getattr(predictor, '_thresholds', None) or {}
                    if thr:
                        thresholds.update({
                            'early_mid': float(thr['early_mid']),
                            'mid_late': float(thr['mid_late']),
                        })
                except Exception:
                    pass
                for food in detected_foods:
                    apply_gas_override_to_food(food, override, thresholds)
                any_spoiled = any(
                    f.get('spoilageLevel') == 'spoiled' for f in detected_foods)
                shrimp_gas_meta = {
                    'applied': True,
                    'level': override.level,
                    'reason': override.reason,
                    'trigger_sensors': override.trigger_sensors,
                    'evidence': override.evidence,
                    'thresholds': thresholds,
                }
                print(f'[shrimp-gas] {override.reason}', flush=True)
            else:
                shrimp_gas_meta = {'applied': False, 'reason': 'no gas rule hit'}
        except Exception as exc:
            print(f'[shrimp-gas] fusion skipped: {exc}', flush=True)
            shrimp_gas_meta = {'applied': False, 'error': str(exc)}

    if not detected_foods:
        top_message = 'Detection Complete. No food items detected.'
    elif any_spoiled:
        top_message = ('Detection Complete. Spoiled items detected — '
                       'please inspect.')
    else:
        top_message = ('Detection Complete. All items analyzed are '
                       'currently fresh.')

    result = {
        'isSpoiled': any_spoiled,
        'message': top_message,
        'detectedFoods': detected_foods,
    }
    if shrimp_gas_meta is not None:
        result['shrimpGasFusion'] = shrimp_gas_meta
    return result


def recognize_foods_ram(image_path, model, image_size, device):
    """Run RAM++ and return (all_tags, food_tags, label_zh)."""
    ram = _ensure_ram()
    transform = ram['get_transform'](image_size=image_size)
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
    tags_en, _tags_cn = ram['inference_ram'](image, model)
    foods = ram['filter_food_tags'](tags_en)
    return tags_en, foods, None


# Backwards-compatible alias used by older callers / detect_server.
def recognize_foods(image_path, model, image_size, device):
    tags, foods, _zh = recognize_foods_ram(image_path, model, image_size, device)
    return tags, foods


def recognize_food_tags(image_path, ram_model=None, image_size=384, device='cpu',
                        tagger='auto', ram_loader=None):
    """Tag foods in an image.

    tagger:
      - 'auto' (default): try vision LLM API first; fall back to RAM++ on any
        failure (missing deps, no key, network/API error, bad JSON, ...).
      - 'vl': vision LLM only (raises on failure).
      - 'ram': RAM++ only (requires ram_model or ram_loader).

    ram_loader: optional zero-arg callable that returns a RAM++ model. Used when
    fallback is needed and ram_model is None (avoids loading RAM if VL succeeds).

    Returns (all_tags, foods, meta) where meta has keys:
      source ('vl'|'ram'), model (optional), label_zh (dict|None).
    """
    tagger = (tagger or 'auto').lower()
    if tagger not in {'auto', 'vl', 'ram'}:
        raise ValueError(f'unknown tagger: {tagger}')

    def _ram():
        nonlocal ram_model
        if ram_model is None:
            if ram_loader is None:
                raise RuntimeError(
                    'RAM++ required but ram_model/ram_loader not provided')
            print('[tagger] Loading RAM++ ...')
            ram_model = ram_loader()
        return ram_model

    if tagger in {'auto', 'vl'}:
        if vl_recognize_foods is None:
            err = 'qwen.recognize_foods import failed (install openai/dotenv?)'
            if tagger == 'vl':
                raise RuntimeError(err)
            print(f'[tagger] VL unavailable ({err}); falling back to RAM++')
        else:
            try:
                result = vl_recognize_foods(image_path)
                foods = list(result.get('foods') or [])
                foods_zh = list(result.get('foods_zh') or [])
                food_counts = {
                    str(k): int(v)
                    for k, v in (result.get('food_counts') or {}).items()
                    if str(k).strip()
                }
                label_zh = {
                    en: zh for en, zh in zip(foods, foods_zh) if zh
                } or None
                # Also map bare noun -> zh for "single banana" / "banana"
                if label_zh:
                    extra = {}
                    for en, zh in label_zh.items():
                        core = en
                        for art in ('a ', 'an ', 'the ', 'single '):
                            if core.startswith(art):
                                core = core[len(art):]
                                break
                        if core and core not in label_zh:
                            extra[core] = zh
                    label_zh.update(extra)
                all_tags = result.get('all_tags') or foods
                if isinstance(all_tags, list):
                    all_tags = ' | '.join(all_tags)
                meta = {
                    'source': 'vl',
                    'model': result.get('model'),
                    'label_zh': label_zh,
                    'food_counts': food_counts,
                }
                print(f"[tagger] VL API ok ({meta.get('model')})")
                if food_counts:
                    print('  expected :',
                          ', '.join(f'{k}×{v}' for k, v in food_counts.items()))
                return all_tags, foods, meta
            except Exception as e:
                if tagger == 'vl':
                    raise
                print(f'[tagger] VL API failed ({e}); falling back to RAM++')

    tags_en, foods, _ = recognize_foods_ram(
        image_path, _ram(), image_size, device)
    return tags_en, foods, {
        'source': 'ram', 'model': 'ram_plus', 'label_zh': None,
        'food_counts': None,
    }


def _load_gdino_image(image_path):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    source = Image.open(image_path).convert('RGB')
    tensor, _ = transform(source, None)
    return source, tensor


def _strip_article(label):
    s = (label or '').lower().strip()
    for art in ('a ', 'an ', 'the ', 'single '):
        if s.startswith(art):
            return s[len(art):].strip()
    return s


def _match_food(phrase, foods):
    """Map a GroundingDINO phrase back to one of the requested food tags."""
    phrase = phrase.lower().strip()
    if not phrase:
        return None
    phrase_core = _strip_article(phrase)

    # Exact / article-insensitive match first.
    for f in foods:
        fl = f.lower().strip()
        if phrase == fl or phrase_core == _strip_article(fl):
            return f

    # Prefer the longest food name contained in / containing the phrase.
    candidates = []
    for f in foods:
        fl = f.lower().strip()
        fc = _strip_article(fl)
        if (fl in phrase or phrase in fl
                or fc in phrase_core or phrase_core in fc):
            candidates.append(f)
    return max(candidates, key=lambda x: len(x)) if candidates else None


def _per_food_limit(food, max_per_food):
    """Resolve how many boxes to keep for one food label."""
    if max_per_food is None:
        return None
    if isinstance(max_per_food, dict):
        if food in max_per_food:
            return max(int(max_per_food[food]), 1)
        # try article-insensitive key
        core = _strip_article(food)
        for k, v in max_per_food.items():
            if _strip_article(k) == core or k.lower() == food.lower():
                return max(int(v), 1)
        return None  # unknown label: keep all
    n = int(max_per_food)
    return n if n > 0 else None


def detect_and_crop(image_path, foods, model, output_dir, device,
                    box_threshold, text_threshold, max_per_food=None):
    """Detect the requested foods, crop each one, and count per food.

    max_per_food:
      - dict {label: n}: keep top-n boxes per label (from VL API counts)
      - int > 0: uniform cap for every label
      - None / 0: keep all
    """
    source, image_tensor = _load_gdino_image(image_path)
    width, height = source.size

    caption = '. '.join(foods)
    boxes, logits, phrases = predict(
        model=model, image=image_tensor, caption=caption,
        box_threshold=box_threshold, text_threshold=text_threshold,
        device=device)

    crops_dir = os.path.join(output_dir, 'crops')
    os.makedirs(crops_dir, exist_ok=True)

    # Collect candidates first so we can rank / truncate per food.
    candidates = []
    scale = torch.tensor([width, height, width, height])
    for box, logit, phrase in zip(boxes, logits, phrases):
        food = _match_food(phrase, foods) or (phrase.strip() or 'unknown')
        cx, cy, bw, bh = (box * scale).tolist()
        x0, y0 = max(int(cx - bw / 2), 0), max(int(cy - bh / 2), 0)
        x1, y1 = min(int(cx + bw / 2), width), min(int(cy + bh / 2), height)
        if x1 <= x0 or y1 <= y0:
            continue
        candidates.append({
            'label': food,
            'probability': round(float(logit), 3),
            'boundingBox': [x0, y0, x1 - x0, y1 - y0],
            'xyxy': (x0, y0, x1, y1),
        })

    if max_per_food:
        by_food = {}
        for c in candidates:
            by_food.setdefault(c['label'], []).append(c)
        kept = []
        for food, items in by_food.items():
            items.sort(key=lambda d: d['probability'], reverse=True)
            limit = _per_food_limit(food, max_per_food)
            kept.extend(items if limit is None else items[:limit])
        candidates = sorted(kept, key=lambda d: d['probability'], reverse=True)

    counts = {}
    detections = []
    for i, c in enumerate(candidates):
        food = c['label']
        counts[food] = counts.get(food, 0) + 1
        crop_name = f'{i:02d}_{food.replace(" ", "_")}.jpg'
        source.crop(c['xyxy']).save(os.path.join(crops_dir, crop_name))
        detections.append({
            'label': food,
            'probability': c['probability'],
            'boundingBox': c['boundingBox'],
            'crop': os.path.join('crops', crop_name),
        })

    return counts, detections


def main():
    parser = argparse.ArgumentParser(
        description='VL/RAM food tags -> GroundingDINO detection/cropping')
    parser.add_argument('--image', required=True, help='path to the photo')
    parser.add_argument('--ram-checkpoint',
                        default='vendor/recognize-anything/pretrained/'
                                'ram_plus_swin_large_14m.pth',
                        help='RAM .pth checkpoint (used on VL failure / --tagger ram)')
    parser.add_argument('--gdino-checkpoint', required=True,
                        help='GroundingDINO .pth checkpoint')
    parser.add_argument('--gdino-config', default=_DEFAULT_GDINO_CFG,
                        help='GroundingDINO config .py')
    parser.add_argument('--image-size', type=int, default=384,
                        help='RAM input image size (default: 384)')
    parser.add_argument('--box-threshold', type=float, default=0.3)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument(
        '--max-per-food', type=int, default=0,
        help='均匀上限：每种食物最多保留几个框（0=不限制）。'
             '若 VL API 已给出每类 count，则优先用 API 的数量；'
             '本参数仅在没有 API count（如 RAM 回退）时生效，或作为强制覆盖。')
    parser.add_argument(
        '--ignore-api-counts', action='store_true',
        help='忽略 VL API 返回的每类数量，改用 --max-per-food')
    parser.add_argument('--output-dir', default='outputs/food_pipeline')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument(
        '--tagger', choices=['auto', 'vl', 'ram'], default='auto',
        help='food tagger: auto=VL API then RAM++, vl=API only, ram=RAM++ only')
    args = parser.parse_args()

    device = 'cpu' if args.cpu_only or not torch.cuda.is_available() else 'cuda'
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Food tagging (VL API preferred; RAM++ loaded only if needed)
    def _load_ram():
        print(f'[1/2] Loading RAM++ on {device} ...')
        ram_plus = _ensure_ram()['ram_plus']
        return ram_plus(pretrained=args.ram_checkpoint,
                        image_size=args.image_size,
                        vit='swin_l').eval().to(device)

    ram_model = _load_ram() if args.tagger == 'ram' else None
    print(f'[1/2] Tagging foods (tagger={args.tagger}) ...')
    all_tags, foods, meta = recognize_food_tags(
        args.image, ram_model=ram_model, image_size=args.image_size,
        device=device, tagger=args.tagger,
        ram_loader=_load_ram if args.tagger == 'auto' else None)
    print('  source   :', meta.get('source'), meta.get('model') or '')
    print('  all tags :', all_tags)
    print('  foods    :', foods if foods else '(none found)')
    if meta.get('food_counts'):
        print('  counts   :',
              ', '.join(f'{k}×{v}' for k, v in meta['food_counts'].items()))

    out_path = os.path.join(args.output_dir, 'detection_results.json')

    if not foods:
        results = build_detection_results(
            args.image, [], label_zh=meta.get('label_zh'),
            output_dir=args.output_dir, device=device)
        results['tagger'] = meta.get('source')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print('No foods detected; skipping GroundingDINO.')
        print(f'Wrote {out_path}')
        return

    # Prefer per-class counts from VL API; fall back to uniform --max-per-food.
    if (not args.ignore_api_counts) and meta.get('food_counts'):
        max_per_food = meta['food_counts']
        print(f'[2/2] GroundingDINO looking for: {foods} '
              f'(limits from API: {max_per_food})')
    else:
        max_per_food = args.max_per_food or None
        print(f'[2/2] GroundingDINO looking for: {foods}'
              + (f' (max_per_food={max_per_food})' if max_per_food else ''))

    gdino_model = load_model(args.gdino_config, args.gdino_checkpoint,
                             device=device)
    counts, detections = detect_and_crop(
        args.image, foods, gdino_model, args.output_dir, device,
        args.box_threshold, args.text_threshold,
        max_per_food=max_per_food)

    print('\nFood counts (amount of food):')
    for food, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f'  {food}: {n}')

    results = build_detection_results(
        args.image, detections, label_zh=meta.get('label_zh'),
        output_dir=args.output_dir, device=device)
    results['tagger'] = meta.get('source')
    if meta.get('food_counts'):
        results['expectedCounts'] = meta['food_counts']
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nCrops + detection_results.json written to {args.output_dir}/')


if __name__ == '__main__':
    main()
