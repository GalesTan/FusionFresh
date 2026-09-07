#!/usr/bin/env python3
"""检测结果可视化：自选 results JSON + crops，按「每个食物」卡片展示。

用法:
  python demo/view_results.py
  python demo/view_results.py --outputs outputs
  python demo/view_results.py --outputs outputs/demo5
  浏览器打开提示的地址。
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, send_file
from urllib.parse import quote

_HERE = Path(__file__).resolve().parents[1]
_INDEX_RE = re.compile(r"^(\d+)_")

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>FusionFresh · 检测结果</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg: #f3efe6;
    --bg-deep: #e7e0d2;
    --ink: #1c241c;
    --muted: #5d675c;
    --line: rgba(28, 36, 28, 0.12);
    --panel: rgba(255, 252, 246, 0.88);
    --fresh: #2f7a4a;
    --fresh-soft: #d7ecd9;
    --warn: #b86b1c;
    --warn-soft: #f5e3c8;
    --spoil: #9b2f3a;
    --spoil-soft: #f3d5d8;
    --spoiling: #c45c18;
    --spoiling-soft: #f7ddc4;
    --accent: #2a5c45;
    --shadow: 0 18px 50px rgba(40, 48, 36, 0.12);
    --radius: 18px;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    font-family: "IBM Plex Sans", sans-serif;
    color: var(--ink);
    background:
      radial-gradient(1200px 600px at 10% -10%, #dff0df 0%, transparent 55%),
      radial-gradient(900px 500px at 100% 0%, #f4dfc8 0%, transparent 50%),
      linear-gradient(180deg, var(--bg), var(--bg-deep));
  }
  body::before {
    content: "";
    position: fixed; inset: 0; pointer-events: none; opacity: 0.35;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.45'/%3E%3C/svg%3E");
    mix-blend-mode: multiply;
  }
  .shell {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 24px;
    max-width: 1440px;
    margin: 0 auto;
    padding: 28px 24px 48px;
    position: relative;
    z-index: 1;
  }
  @media (max-width: 960px) {
    .shell { grid-template-columns: 1fr; }
  }
  .brand {
    font-family: Fraunces, Georgia, serif;
    font-size: clamp(1.7rem, 2.4vw, 2.2rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 6px;
  }
  .brand span { color: var(--accent); }
  .sub { color: var(--muted); margin: 0 0 18px; font-size: 0.95rem; line-height: 1.45; }
  .panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
    padding: 18px;
  }
  .side { align-self: start; position: sticky; top: 18px; }
  @media (max-width: 960px) { .side { position: static; } }
  label {
    display: block;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 14px 0 6px;
  }
  label:first-of-type { margin-top: 0; }
  select, input[type="text"] {
    width: 100%;
    border: 1px solid var(--line);
    background: #fffdf8;
    border-radius: 12px;
    padding: 10px 12px;
    font: inherit;
    color: var(--ink);
  }
  select:focus, input:focus {
    outline: 2px solid rgba(42, 92, 69, 0.35);
    border-color: var(--accent);
  }
  .row { display: flex; gap: 8px; }
  .row input { flex: 1; }
  button {
    border: 0;
    border-radius: 999px;
    padding: 10px 16px;
    font: inherit;
    font-weight: 600;
    cursor: pointer;
    background: var(--accent);
    color: #f7fff8;
    transition: transform .15s ease, filter .15s ease;
  }
  button:hover { transform: translateY(-1px); filter: brightness(1.05); }
  button.ghost {
    background: transparent;
    color: var(--accent);
    border: 1px solid rgba(42, 92, 69, 0.35);
  }
  .hint {
    margin-top: 12px;
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.4;
  }
  .summary {
    display: grid;
    grid-template-columns: 1.4fr auto;
    gap: 16px;
    align-items: stretch;
    margin-bottom: 18px;
    animation: rise .45s ease both;
  }
  @media (max-width: 720px) { .summary { grid-template-columns: 1fr; } }
  .summary h2 {
    font-family: Fraunces, Georgia, serif;
    margin: 0 0 8px;
    font-size: 1.55rem;
  }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 999px;
    background: #fff;
    border: 1px solid var(--line);
    font-size: 0.85rem;
  }
  .status {
    min-width: 160px;
    border-radius: 16px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
  }
  .status.ok { background: var(--fresh-soft); color: var(--fresh); }
  .status.warn { background: var(--spoiling-soft); color: var(--spoiling); }
  .status.bad { background: var(--spoil-soft); color: var(--spoil); }
  .status .big {
    font-family: Fraunces, Georgia, serif;
    font-size: 1.4rem;
    font-weight: 700;
  }
  .filters {
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px;
  }
  .filters button {
    padding: 7px 12px;
    font-size: 0.85rem;
    background: #fffdf8;
    color: var(--ink);
    border: 1px solid var(--line);
  }
  .filters button.active {
    background: var(--accent);
    color: #f7fff8;
    border-color: transparent;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 16px;
  }
  .card {
    background: #fffdf9;
    border: 1px solid var(--line);
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 10px 28px rgba(40, 48, 36, 0.08);
    display: flex;
    flex-direction: column;
    animation: rise .5s ease both;
    transition: transform .18s ease, box-shadow .18s ease;
  }
  .card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 36px rgba(40, 48, 36, 0.14);
  }
  .card.level-fresh {
    background: linear-gradient(180deg, #f4fbf4, #fffdf9 42%);
    border-color: rgba(47, 122, 74, 0.22);
  }
  .card.level-spoiling {
    background: linear-gradient(180deg, #fff4e8, #fffdf9 42%);
    border-color: rgba(196, 92, 24, 0.28);
  }
  .card.level-spoiled {
    background: linear-gradient(180deg, #fceced, #fffdf9 42%);
    border-color: rgba(155, 47, 58, 0.28);
  }
  .card .thumb {
    aspect-ratio: 1;
    background:
      linear-gradient(45deg, #ece7dc 25%, transparent 25%) -12px 0 / 24px 24px,
      linear-gradient(-45deg, #ece7dc 25%, transparent 25%) -12px 0 / 24px 24px,
      #f7f2e8;
    display: grid;
    place-items: center;
    overflow: hidden;
  }
  .card .thumb img {
    width: 100%; height: 100%; object-fit: contain;
    transition: transform .35s ease;
  }
  .card:hover .thumb img { transform: scale(1.04); }
  .card .body { padding: 14px 14px 16px; display: flex; flex-direction: column; gap: 8px; flex: 1; }
  .card .top {
    display: flex; justify-content: space-between; align-items: baseline; gap: 8px;
  }
  .card .title {
    font-family: Fraunces, Georgia, serif;
    font-size: 1.2rem;
    margin: 0;
  }
  .idx { color: var(--muted); font-size: 0.85rem; font-weight: 600; }
  .badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 9px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .badge.fresh { background: var(--fresh-soft); color: var(--fresh); }
  .badge.spoiling { background: var(--spoiling-soft); color: var(--spoiling); }
  .badge.spoiled { background: var(--spoil-soft); color: var(--spoil); }
  .badge.unknown { background: #ebe6dc; color: var(--muted); }
  .meter {
    height: 8px; border-radius: 999px; background: #ebe4d7; overflow: hidden;
  }
  .meter > i {
    display: block; height: 100%; border-radius: inherit;
    background: linear-gradient(90deg, var(--fresh), var(--spoiling), var(--spoil));
  }
  .meta { font-size: 0.86rem; color: var(--muted); line-height: 1.45; }
  .gases { display: flex; flex-wrap: wrap; gap: 6px; }
  .gas {
    font-size: 0.75rem;
    padding: 3px 8px;
    border-radius: 999px;
    background: #eef3ea;
    color: #35513d;
  }
  .empty {
    padding: 48px 20px;
    text-align: center;
    color: var(--muted);
    border: 1px dashed var(--line);
    border-radius: var(--radius);
    background: rgba(255,252,246,.55);
  }
  @keyframes rise {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: none; }
  }
</style>
</head>
<body>
  <div class="shell">
    <aside class="panel side">
      <h1 class="brand">Fusion<span>Fresh</span></h1>
      <p class="sub">选择一次检测输出，按「每个食物」查看裁剪图与腐败评分。</p>

      <label for="run">输出目录</label>
      <select id="run"></select>

      <label for="results">results JSON</label>
      <select id="results"></select>

      <label for="crops">crops 目录</label>
      <div class="row">
        <input id="crops" type="text" placeholder="相对或绝对路径" />
      </div>

      <div class="row" style="margin-top:14px">
        <button id="loadBtn" style="flex:1">加载</button>
        <button id="refreshBtn" class="ghost" type="button">刷新列表</button>
      </div>
      <p class="hint" id="pathHint">默认扫描 {{ outputs }}</p>
    </aside>

    <main>
      <section class="panel summary" id="summary" hidden>
        <div>
          <h2 id="sumTitle">—</h2>
          <div class="meta" id="sumMsg"></div>
          <div class="chips" id="sumChips"></div>
        </div>
        <div class="status" id="sumStatus">
          <div class="big" id="sumStatusText">—</div>
          <div id="sumCount"></div>
        </div>
      </section>

      <div class="filters" id="filters" hidden>
        <button data-filter="all" class="active">全部</button>
        <button data-filter="fresh">新鲜</button>
        <button data-filter="spoiling">转差中</button>
        <button data-filter="spoiled">已腐败</button>
      </div>

      <div class="grid" id="grid"></div>
      <div class="empty" id="empty">在左侧选择一次检测结果并点击「加载」。</div>
    </main>
  </div>

<script>
const $ = (id) => document.getElementById(id);
let currentFoods = [];
let activeFilter = "all";

function levelClass(level) {
  const l = (level || "").toLowerCase().trim();
  if (l === "spoiled" || l.includes("inedible") || l.includes("rotten")) return "spoiled";
  if (l === "spoiling" || l.includes("uncertain") || l.includes("ripe") || l.includes("warn")) return "spoiling";
  if (l === "fresh" || l.includes("fresh")) return "fresh";
  return "unknown";
}

function levelLabelZh(level) {
  const cls = levelClass(level);
  if (cls === "fresh") return "新鲜";
  if (cls === "spoiling") return "转差中";
  if (cls === "spoiled") return "已腐败";
  return level || "未知";
}

function pct(v) {
  if (v == null || Number.isNaN(Number(v))) return null;
  return Math.max(0, Math.min(1, Number(v)));
}

async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function refreshRuns() {
  const data = await api("/api/runs");
  const sel = $("run");
  const prev = sel.value;
  sel.innerHTML = "";
  data.runs.forEach((run) => {
    const opt = document.createElement("option");
    opt.value = run.id;
    opt.textContent = `${run.id}  ·  ${run.n_foods} foods · ${run.n_crops} crops`;
    sel.appendChild(opt);
  });
  if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
  await onRunChange();
}

async function onRunChange() {
  const run = $("run").value;
  if (!run) return;
  const data = await api(`/api/runs/${encodeURIComponent(run)}`);
  const rsel = $("results");
  rsel.innerHTML = "";
  data.results_files.forEach((f, i) => {
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = f;
    if (i === 0) opt.selected = true;
    rsel.appendChild(opt);
  });
  $("crops").value = data.crops_dir || "";
  $("pathHint").textContent = data.path;
}

function render() {
  const grid = $("grid");
  const empty = $("empty");
  grid.innerHTML = "";
  const foods = currentFoods.filter((f) => {
    if (activeFilter === "all") return true;
    return levelClass(f.spoilageLevel) === activeFilter;
  });
  if (!foods.length) {
    empty.hidden = false;
    empty.textContent = currentFoods.length ? "当前筛选下没有食物。" : "没有可展示的食物。";
    return;
  }
  empty.hidden = true;
  foods.forEach((f, i) => {
    const score = pct(f.spoilageScore);
    const cls = levelClass(f.spoilageLevel);
    const card = document.createElement("article");
    card.className = `card level-${cls}`;
    card.style.animationDelay = `${Math.min(i, 12) * 0.03}s`;
    const img = f.crop_url
      ? `<img src="${f.crop_url}" alt="${f.label || "食物"}" loading="lazy" />`
      : `<div class="meta">无裁剪图</div>`;
        const gases = (f.producedGases || []).map(g => {
      const zh = ({
        "VOC": "挥发性有机物",
        "Ethanol": "乙醇",
        "Ammonia": "氨气",
        "Hydrogen Sulfide": "硫化氢",
        "Ethylene": "乙烯",
        "Methanethiol": "甲硫醇",
      })[g] || g;
      return `<span class="gas">${zh}</span>`;
    }).join("");
    card.innerHTML = `
      <div class="thumb">${img}</div>
      <div class="body">
        <div class="top">
          <h3 class="title">${f.label || "未知"}</h3>
          <span class="idx">#${String(f.index).padStart(2, "0")}</span>
        </div>
        <div>
          <span class="badge ${cls}">${levelLabelZh(f.spoilageLevel)}</span>
        </div>
        <div class="meta">参考分 ${score == null ? "—" : score.toFixed(2)}</div>
        <div class="meter"><i style="width:${score == null ? 0 : score * 100}%"></i></div>
        <div class="meta">${f.message || ""}</div>
        ${gases ? `<div class="gases">${gases}</div>` : ""}
      </div>`;
    grid.appendChild(card);
  });
}

async function loadSelected() {
  const run = $("run").value;
  const results = $("results").value;
  const crops = $("crops").value.trim();
  const q = new URLSearchParams({ run, results, crops });
  const data = await api(`/api/load?${q}`);
  currentFoods = data.foods || [];

  $("summary").hidden = false;
  $("filters").hidden = false;
  $("sumTitle").textContent = data.run_id;
  $("sumMsg").textContent = data.message || "";
  const levels = currentFoods.map(f => levelClass(f.spoilageLevel));
  const hasSpoiled = levels.includes("spoiled") || data.isSpoiled;
  const hasSpoiling = levels.includes("spoiling");
  $("sumStatus").className = "status " + (hasSpoiled ? "bad" : (hasSpoiling ? "warn" : "ok"));
  $("sumStatusText").textContent = hasSpoiled ? "不宜食用" : (hasSpoiling ? "建议尽快食用" : "状态良好");
  $("sumCount").textContent = `${currentFoods.length} 件`;

  const chips = $("sumChips");
  chips.innerHTML = "";
  if (data.shrimpGasFusion && data.shrimpGasFusion.applied) {
    const g = document.createElement("span");
    g.className = "chip";
    g.style.background = "#f7ddc4";
    g.textContent = "已结合气体检测";
    chips.appendChild(g);
  }

  render();
}

$("run").addEventListener("change", () => onRunChange().catch(alert));
$("loadBtn").addEventListener("click", () => loadSelected().catch(alert));
$("refreshBtn").addEventListener("click", () => refreshRuns().catch(alert));
$("filters").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-filter]");
  if (!btn) return;
  activeFilter = btn.dataset.filter;
  [...$("filters").querySelectorAll("button")].forEach(b => b.classList.toggle("active", b === btn));
  render();
});

refreshRuns().catch(alert);
</script>
</body>
</html>
"""


def _safe_under(root: Path, target: Path) -> Path:
    root = root.resolve()
    target = target.resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"path escapes root: {target}")
    return target


def _run_info(d: Path, run_id: str) -> dict | None:
    jsons = sorted(d.glob("detection_results*.json"))
    if not jsons:
        return None
    crops = d / "crops"
    n_crops = 0
    if crops.is_dir():
        n_crops = len(list(crops.glob("*.jpg"))) + len(list(crops.glob("*.png")))
    n_foods = 0
    try:
        data = json.loads(jsons[0].read_text(encoding="utf-8"))
        n_foods = len(data.get("detectedFoods") or [])
    except Exception:
        pass
    return {
        "id": run_id,
        "path": str(d),
        "n_foods": n_foods,
        "n_crops": n_crops,
        "results_files": [p.name for p in jsons],
    }


def discover_runs(outputs: Path) -> list[dict]:
    runs = []
    if not outputs.is_dir():
        return runs

    # --outputs 指向单个 run 目录（本身含 detection_results*.json）
    self_run = _run_info(outputs, ".")
    if self_run:
        self_run["id"] = outputs.name
        self_run["_single"] = True
        runs.append(self_run)
        return runs

    for d in sorted(outputs.iterdir(), key=lambda p: p.name.lower()):
        if not d.is_dir():
            continue
        info = _run_info(d, d.name)
        if info:
            runs.append(info)
    return runs


def resolve_run_dir(outputs: Path, run_id: str) -> Path:
    """Resolve a run id to its directory.

    Supports:
      - parent outputs/ + child id  (e.g. outputs + demo_latest)
      - --outputs pointed at the run itself (id == folder name)
    """
    outputs = outputs.resolve()
    # Single-run mode: --outputs is the run folder.
    if any(outputs.glob("detection_results*.json")):
        if run_id in (".", outputs.name):
            return outputs
        # still allow only this folder
        raise FileNotFoundError(run_id)
    return _safe_under(outputs, outputs / run_id)


def list_crop_files(crops_dir: Path) -> list[Path]:
    if not crops_dir.is_dir():
        return []
    files = list(crops_dir.glob("*.jpg")) + list(crops_dir.glob("*.png")) + list(crops_dir.glob("*.jpeg"))
    return sorted(files, key=lambda p: p.name.lower())


def match_crop(food: dict, index: int, crops_dir: Path, crop_files: list[Path]) -> Path | None:
    rel = food.get("crop")
    if rel:
        cand = Path(rel)
        if not cand.is_absolute():
            cand = crops_dir.parent / rel if (crops_dir.parent / rel).is_file() else crops_dir / cand.name
        if cand.is_file():
            return cand

    prefix = f"{index:02d}_"
    for p in crop_files:
        if p.name.startswith(prefix):
            return p
    # loose: numeric prefix match without zero-pad
    for p in crop_files:
        m = _INDEX_RE.match(p.name)
        if m and int(m.group(1)) == index:
            return p
    if 0 <= index < len(crop_files):
        return crop_files[index]
    return None


def build_app(outputs: Path) -> Flask:
    app = Flask(__name__)
    outputs = outputs.resolve()

    @app.get("/")
    def index():
        return render_template_string(HTML, outputs=str(outputs))

    @app.get("/api/runs")
    def api_runs():
        return jsonify({"outputs": str(outputs), "runs": discover_runs(outputs)})

    @app.get("/api/runs/<run_id>")
    def api_run(run_id: str):
        try:
            run_dir = resolve_run_dir(outputs, run_id)
        except (ValueError, FileNotFoundError):
            abort(404)
        if not run_dir.is_dir():
            abort(404)
        jsons = sorted(run_dir.glob("detection_results*.json"))
        crops = run_dir / "crops"
        return jsonify({
            "id": run_id,
            "path": str(run_dir),
            "results_files": [p.name for p in jsons],
            "crops_dir": str(crops) if crops.is_dir() else "",
        })

    @app.get("/api/load")
    def api_load():
        run_id = request.args.get("run", "").strip()
        results_name = request.args.get("results", "detection_results.json").strip()
        crops_arg = request.args.get("crops", "").strip()

        if not run_id:
            abort(400, "run required")
        try:
            run_dir = resolve_run_dir(outputs, run_id)
        except (ValueError, FileNotFoundError):
            abort(404, f"run not found: {run_id}")
        results_path = run_dir / results_name
        if not results_path.is_file():
            # allow absolute / relative override if still under outputs or absolute existing file
            alt = Path(results_name)
            if alt.is_file():
                results_path = alt
            else:
                abort(404, f"results not found: {results_name}")

        if crops_arg:
            crops_dir = Path(crops_arg)
            if not crops_dir.is_absolute():
                crops_dir = (run_dir / crops_arg).resolve() if (run_dir / crops_arg).exists() else (outputs / crops_arg).resolve()
            if not crops_dir.is_dir():
                abort(404, f"crops dir not found: {crops_arg}")
        else:
            crops_dir = run_dir / "crops"

        data = json.loads(results_path.read_text(encoding="utf-8"))
        foods_raw = data.get("detectedFoods") or []
        crop_files = list_crop_files(crops_dir)
        default_crops = (run_dir / "crops").resolve()
        crops_resolved = crops_dir.resolve()
        use_default = crops_resolved == default_crops

        foods = []
        for i, food in enumerate(foods_raw):
            crop_path = match_crop(food, i, crops_dir, crop_files)
            crop_url = None
            if crop_path is not None:
                if use_default:
                    crop_url = f"/crop/{quote(run_id)}/{quote(crop_path.name)}"
                else:
                    crop_url = (
                        f"/cropfile?path={quote(str(crop_path.resolve()))}"
                    )
            item = {
                "index": i,
                "label": food.get("label"),
                "type_probability": food.get("type_probability"),
                "spoilageScore": food.get("spoilageScore"),
                "spoilageLevel": food.get("spoilageLevel"),
                "boundingBox": food.get("boundingBox"),
                "message": food.get("message"),
                "producedGases": food.get("producedGases") or [],
                "source_image": food.get("source_image"),
                "gasOverride": food.get("gasOverride"),
                "imageModel": food.get("imageModel"),
                "crop_file": crop_path.name if crop_path else None,
                "crop_url": crop_url,
            }
            foods.append(item)

        return jsonify({
            "run_id": run_id,
            "results_file": results_path.name,
            "crops_dir": str(crops_dir),
            "isSpoiled": bool(data.get("isSpoiled")),
            "message": data.get("message"),
            "tagger": data.get("tagger"),
            "expectedCounts": data.get("expectedCounts") or {},
            "shrimpGasFusion": data.get("shrimpGasFusion"),
            "foods": foods,
        })

    @app.get("/crop/<run_id>/<path:filename>")
    def crop_in_run(run_id: str, filename: str):
        try:
            run_dir = resolve_run_dir(outputs, run_id)
        except (ValueError, FileNotFoundError):
            abort(404)
        name = Path(filename).name
        path = (run_dir / "crops" / name).resolve()
        try:
            _safe_under(run_dir / "crops", path)
        except ValueError:
            abort(404)
        if not path.is_file():
            abort(404)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return send_file(path, mimetype=mime)

    @app.get("/cropfile")
    def crop_file():
        """Serve an absolute crop path that still lives under --outputs."""
        raw = request.args.get("path", "").strip()
        if not raw:
            abort(400)
        path = Path(raw).resolve()
        try:
            _safe_under(outputs, path)
        except ValueError:
            abort(403)
        if not path.is_file():
            abort(404)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return send_file(path, mimetype=mime)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="可视化 outputs 检测结果（按食物单位）")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=_HERE / "outputs",
        help="outputs 根目录（可浏览全部 run），或单个 run 目录（如 outputs/demo_latest）",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    outputs = args.outputs.resolve()
    if not outputs.is_dir():
        raise SystemExit(f"outputs 目录不存在: {outputs}")

    app = build_app(outputs)
    print(f"Outputs: {outputs}")
    print(f"Open:    http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
