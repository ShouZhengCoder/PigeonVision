# PigeonVision — Codex Agent Context

## Project

Pigeon iris recognition system. Two functions:
1. **Compare**: given two iris images, output Euclidean distance + same-breed judgment
2. **Search**: given one iris image, return Top-K similar pigeons with PG_ID and breed name

Full spec: read `ROADMAP.md` before writing any code.

## Absolute Project Root

```
/home/u2023312335/project/learn/PigeonVision
```

## Key Paths

| Path | Content |
|------|---------|
| `data/extracted/{1..12}/` | Raw pigeon eye images, `{img_id}.jpg` directly inside each numbered dir |
| `data/extracted/datasetXGN/anotations/` | 9,979 JSON annotation files (eye bounding boxes) |
| `data/extracted/datasetXGN/relations.csv` | 250,207 (blood_id, img_id) pairs — **no header row**, use `header=None, names=['blood_id','img_id']`; primary source for pair building |
| `data/extracted/datasetXGN/blood.csv` | **DEPRECATED** — wide-format bloodline table; all data is a subset of relations.csv; do not use in new scripts |
| `data/extracted/datasetXGN/pigeon.csv` | 113,844 records with ID, PG_ID, BLOOD, EYE fields |
| `data/extracted/datasetXGN/img_list.txt` | 31,900 image IDs, one per line |
| `data/unet_labelme_80/` | 80 manually labeled iris samples for U-Net v1 |
| `outputs/img_index.csv` | **Built in Stage 1**: img_id → absolute file path (all scripts use this) |
| `outputs/eye_crops/` | Eye region crops from YOLO inference |
| `outputs/iris_normalized/` | 64×512 normalized iris images |
| `outputs/features/` | Feature vectors + FAISS index |
| `checkpoints/detection/` | YOLOv5 weights |
| `checkpoints/segmentation/` | U-Net weights |
| `checkpoints/siamese/` | Siamese encoder weights |
| `configs/unet.yaml` | U-Net training config |
| `configs/` | Training configs (YAML) |
| `src/stage{1..6}_*/` | Source code per stage |

## Image Lookup Rule

Images are in `data/extracted/1/` through `data/extracted/12/`. To find a given `img_id`, always use the pre-built index:

```python
import pandas as pd
def load_img_index():
    df = pd.read_csv("outputs/img_index.csv")
    return dict(zip(df["img_id"].astype(str), df["path"]))
```

**Never hardcode a subdirectory number.** Always query the index.

## Data Gotchas

- `relations.csv` has **no header row** — always read with `header=None, names=['blood_id','img_id']`; do NOT use `blood.csv` (deprecated)
- Annotation JSONs contain labels `"eye"`, `"mouse"`, `"900"` → **keep only `"eye"`**
- `pigeon.csv` has 11 columns: `ID, PID, CID, SID, NAME, COLOR, EYE, PG_ID, SEX, BLOOD, IMG`; field `weidth` in annotation JSONs is a typo for `width` — use as-is
- MobileNetV2 expects 3-channel RGB input; grayscale iris images need `.convert("RGB")` before resize

## Stage Status

Update this section as stages complete:

- [ ] Stage 1 — Data prep (`outputs/img_index.csv`, `data/yolo_dataset/`, `data/pairs_*.csv`)
- [ ] Stage 2 — Eye detection (`checkpoints/detection/best.pt`, `outputs/eye_crops/`)
- [x] Stage 3 — Iris segmentation + normalization (`outputs/iris_normalized/`)
- [x] Stage 3.5 — Rebuild pairs (`data/pairs_train.csv` rebuilt from full iris_normalized set)
- [x] Stage 4 — Siamese training (`checkpoints/siamese/best.pt`, `outputs/features/`)
- [x] Stage 5 — Flask server (`src/stage5_server/app.py`)
- [ ] Stage 6 — Android deploy (`src/stage6_android/`)

## Critical Rules

1. Stage 3.5 (rebuild_pairs) MUST run before Stage 4 training
2. Stage 3 training/inference MUST use the same `256x256`, 1-channel UNet setting with `GroupNorm(num_groups=8)` when `base_c=32`
3. build_db.py MUST cross-reference normalize_meta.csv (status=success) before extracting features
4. convert_annotations.py MUST skip images absent from img_index.csv and report count
5. All batch scripts MUST support --resume to skip already-processed items

## Git Conventions

**Commit message format**: `<type>(<scope>): <中文说明>`

| type | 用途 |
|------|------|
| `init` | 仓库初始化 |
| `feat` | 新增功能或脚本 |
| `fix` | 修复 bug |
| `refactor` | 重构，不改变功能 |
| `docs` | 文档更新 |
| `chore` | 配置、依赖变更 |

示例：
- `feat(stage1): 数据整理脚本，图片索引、YOLO标注转换、样本对构建`
- `fix(stage3): 替换为 U-Net 分割 + 椭圆展开，修复归一化成功率过低的问题`
- `docs: 更新 ROADMAP.md 阶段三说明`

**不应进入 git 的文件**（.gitignore 已配置）：
- 原始图片 `data/extracted/`
- 生成的图片 `outputs/eye_crops/`, `outputs/iris_normalized/*.png`
- 模型权重 `checkpoints/`
- 大型二进制 `outputs/features/*.npy`, `*.bin`
- 训练日志 `logs/`

**应该进入 git 的文件**：
- 所有 `src/` 源代码
- `configs/` 配置文件
- CSV 元数据（crop_meta.csv, normalize_meta.csv, feature_db_meta.csv, pairs_*.csv）
- `data/yolo_dataset/data.yaml`
- `ROADMAP.md`, `AGENTS.md`, `AGENT_PROMPTS.md`
- `requirements.txt`

每完成一个 Stage 必须提交一次，提交前先 `git status` 确认没有大文件混入。

## Coding Conventions

- All scripts accept CLI args via `argparse`; paths default to project-root-relative values
- Use `tqdm` for any loop over >100 items
- Support `--resume` flag on batch processing scripts (skip already-processed items)
- GPU/CPU auto-detection: `device = "cuda" if torch.cuda.is_available() else "cpu"`
- Log progress and final stats to stdout; errors to stderr
- No hardcoded absolute paths in source files — derive from project root or pass as args
