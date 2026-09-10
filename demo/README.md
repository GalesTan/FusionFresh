# Demo

用于在没有采集硬件时，走与线上相同的 `food_pipeline.py`。

## 示例图

| 文件 | 说明（按文件名，内容以实图为准） |
|------|----------------------------------|
| `demo5.jpg` … `demo12.jpg` | 原项目 `demo/` 中的联调图 |

## 跑一次检测

在仓库根目录：

```powershell
python demo/run_demo.py --image demo/demo16.jpg --cpu-only
```

或在 `demo/` 目录下用文件名：

```powershell
python run_demo.py --image demo13.jpg --cpu-only
```

结果写到 `outputs/demo_demo6/`（含 `detection_results.json` 与 `crops/`）。

等价于直接调用流水线：

```powershell
python food_pipeline.py --image demo/demo6.jpg --gdino-checkpoint vendor/GroundingDINO/weights/groundingdino_swint_ogc.pth --output-dir outputs/demo_demo6 --cpu-only
```

需要 Qwen VL 时请先配置 `qwen/.env`。`--tagger ram` 可强制只用 RAM++（不调 API）。

## 看结果

```powershell
python demo/view_results.py --outputs outputs
```

浏览器打开终端打印的地址，按「每个食物」卡片查看框选与评分。
