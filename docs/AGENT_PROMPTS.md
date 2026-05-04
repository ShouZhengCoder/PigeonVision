# Codex CLI 指令手册

## 使用方式

```bash
cd /home/u2023312335/project/learn/PigeonVision
codex   # 启动后 Codex 自动读取 AGENTS.md
```

将下面对应阶段的指令复制后粘贴给 Codex。**每次只执行一个阶段，验收通过后再执行下一个。**

---

## Stage 0：Git 初始化

```
在项目根目录 /home/u2023312335/project/learn/PigeonVision 下完成 Git 初始化。

【第一步】初始化仓库
git init
git config user.name "你的名字"
git config user.email "你的邮箱"

【第二步】创建 .gitignore
在项目根目录创建 .gitignore，内容如下：

# 原始图片（体积太大，不入库）
data/extracted/

# 生成的中间产物（可由代码重新生成）
outputs/eye_crops/*.jpg
outputs/iris_normalized/*.png

# 模型权重（体积大，用训练脚本重现）
checkpoints/

# 大型二进制特征文件
outputs/features/*.npy
outputs/features/*.bin

# 训练日志
logs/

# Python 缓存
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# 编辑器
.vscode/
.idea/
*.swp

# 系统文件
.DS_Store
Thumbs.db

# 虚拟环境
venv/
.env

# YOLOv5 训练产物（含大量图片缓存）
data/yolo_dataset/labels/
data/yolo_dataset/train.txt
data/yolo_dataset/val.txt

注意：以下文件需要追踪（不在 .gitignore 里）：
- src/**（所有源代码）
- configs/**（训练配置）
- data/yolo_dataset/data.yaml（数据集配置）
- data/datasetXGN/relations.csv、pigeon.csv、img_list.txt 等元数据 CSV（blood.csv 已弃用但可保留）
- outputs/eye_crops/crop_meta.csv（检测结果元数据，小文件）
- outputs/iris_normalized/normalize_meta.csv（预处理结果元数据）
- outputs/features/feature_db_meta.csv（特征库元数据）
- ROADMAP.md、AGENTS.md、AGENT_PROMPTS.md
- requirements.txt

【第三步】首次提交
git add .
git status   # 确认暂存的文件符合预期，大文件不在其中
git commit -m "init: 项目初始化，添加文档和 .gitignore"

【第四步】确认
运行 git log --oneline 和 git status，确认仓库状态干净。
打印当前追踪的文件列表（git ls-files）。
```

---

## Stage 1：数据整理

```
请先完整阅读 ROADMAP.md，然后实现以下三个脚本。

【脚本一】src/stage1_data/build_img_index.py
遍历 data/extracted/1/ 到 data/extracted/12/，收集所有 .jpg/.jpeg 文件。
img_id 为文件名去掉后缀，path 为绝对路径。
写入 outputs/img_index.csv（列：img_id, path）。
运行完打印总图片数。

【脚本二】src/stage1_data/convert_annotations.py
读取 data/datasetXGN/anotations/ 下所有 JSON，只保留 label=="eye" 的标注框，过滤掉 "mouse" 和 "900"。
先查 outputs/img_index.csv，跳过图片文件不存在的标注（打印跳过数量）。
将 bbx [x1,y1,x2,y2] 转换为 YOLO 格式 [0 cx cy w h]（归一化到0~1，尺寸从 JSON 的 height/weidth 字段读取）。
按 8:2 随机分割（seed=42），将 YOLO 格式 .txt 写入 data/yolo_dataset/labels/train/ 和 /val/。
生成 data/yolo_dataset/train.txt 和 val.txt（每行一个图片绝对路径）。
生成 data/yolo_dataset/data.yaml（nc=1, names:['eye'], train/val 字段指向 train.txt/val.txt 的绝对路径）。
运行完打印：处理图片数、跳过数、train/val 各多少张。

【脚本三】src/stage1_data/build_pairs.py
读取 data/datasetXGN/relations.csv，注意该文件无 header 行，必须用：
  rel = pd.read_csv('data/datasetXGN/relations.csv', header=None, names=['blood_id', 'img_id'])
  rel['img_id'] = rel['img_id'].astype(str)
不要使用 blood.csv（已弃用）。
获取 data/datasetXGN/anotations/ 目录下所有标注图片的 ID 集合（valid_imgs）。
将 relations.csv 过滤到 valid_imgs：rel = rel[rel['img_id'].isin(valid_imgs)]
正样本对：同一 blood_id 下所有两两组合（itertools.combinations），label=1。
  注意：同一对图片可能共属多个 blood_id，必须去重：pairs_pos = list(set(pairs_pos))
负样本对：跨 blood_id 随机采样，数量 = 正样本数 × 2，seed=42，label=0。
  负样本同样需去重，且确保 (a,b) 和 (b,a) 视为同一对。
按 8:2 分割，写入 data/pairs_train.csv 和 data/pairs_val.csv（列：img_id_a, img_id_b, label）。
运行完打印：正样本对数、负样本对数、train/val 各多少行。

每个脚本写完后立即运行验证，修复所有报错。
完成后执行：git add src/stage1_data/ data/yolo_dataset/data.yaml && git commit -m "feat(stage1): 数据整理脚本，图片索引、YOLO标注转换、样本对构建"
```

---

## Stage 2：目标检测训练

```
请先完整阅读 ROADMAP.md，然后实现以下内容。前提：Stage 1 已完成。

【脚本一】src/stage2_detection/train.py
封装 YOLOv5 训练命令，使用 ultralytics 库。
训练参数：data=data/yolo_dataset/data.yaml, model=yolov5s.pt, epochs=100, batch=16, imgsz=416。
权重保存到 checkpoints/detection/。
支持 --epochs、--batch 命令行参数覆盖默认值。
打印实际执行的命令后再运行。

【脚本二】src/stage2_detection/infer_all.py
加载 checkpoints/detection/ 下最新的 best.pt。
从 outputs/img_index.csv 读取所有图片路径，遍历推理。
每张图取置信度最高的 "eye" 框（阈值 0.7），无检测结果则记录 confidence=0。
有检测结果的图：bbox 四边各扩展 10%（不超出图片边界），裁剪保存到 outputs/eye_crops/{img_id}.jpg。
将所有记录写入 outputs/eye_crops/crop_meta.csv（列：img_id, x1, y1, x2, y2, confidence）。
支持 --resume 参数，跳过 crop_meta.csv 中已有记录的图片。
用 tqdm 显示进度，每处理 1000 张打印一次检出率。

先运行 train.py 确认训练正常启动，打印第一个 epoch 的日志。
训练完成后再运行 infer_all.py。
完成后执行：git add src/stage2_detection/ configs/yolov5.yaml && git commit -m "feat(stage2): YOLOv5训练脚本和全量推理脚本"
```

---

## Stage 3：虹膜图像预处理

```
请先完整阅读 ROADMAP.md，然后实现以下内容。前提：Stage 2 已完成，outputs/eye_crops/ 下有图片。

【脚本一】src/stage3_preprocess/iris_localize.py
实现函数 localize_iris(img_bgr) -> (cx, cy, r_inner, r_outer) 或 None（定位失败返回 None）。
算法：灰度化 → 高斯模糊(5×5) → 水平和垂直方向一维投影取最小值，确定瞳孔粗略中心 (cx,cy)
→ 以 (cx,cy) 为中心取 ROI → Canny(低阈值50, 高阈值150) → cv2.HoughCircles 检测内外圆。
内圆半径范围：图像宽度的 10%~25%；外圆半径范围：30%~50%。
HoughCircles 参数：method=HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30。

实现函数 normalize_iris(img_bgr, cx, cy, r_inner, r_outer, shape=(64,512)) -> 64×512 灰度 numpy 数组。
算法（Daugman 极坐标展开）：角度方向 512 步（0 到 2π），径向 64 步（从内圆到外圆），
每个采样点用双线性插值取像素值。

实现函数 visualize_localization(img_bgr, cx, cy, r_inner, r_outer) -> 标注图（画出两个圆）。

【脚本二】src/stage3_preprocess/batch_normalize.py
读取 outputs/eye_crops/crop_meta.csv，处理所有 confidence > 0 的图片。
对每张图调用 localize_iris + normalize_iris。
成功：保存 64×512 灰度 PNG 到 outputs/iris_normalized/{img_id}.png。
失败：记录 status=failed，不保存图片。
写入 outputs/iris_normalized/normalize_meta.csv（列：img_id, status, cx, cy, r_inner, r_outer）。
支持 --resume，跳过 normalize_meta.csv 中已有记录的图片。
每处理 1000 张打印一次当前成功率。

【脚本三】src/stage3_preprocess/visualize_samples.py
从 normalize_meta.csv 中随机抽取 20 张 status=success 的图片。
每张图左边显示 eye_crop 原图，右边显示归一化后的 64×512 虹膜图。
将 20 对图片拼成网格，保存到 outputs/iris_normalized/samples_vis.png。

先用 50 张图片小批量测试 batch_normalize.py，确认能正常产出 PNG 文件后再全量运行。
全量完成后运行 visualize_samples.py 生成对比图。
完成后执行：git add src/stage3_preprocess/ outputs/iris_normalized/normalize_meta.csv outputs/iris_normalized/samples_vis.png && git commit -m "feat(stage3): 虹膜定位归一化脚本，成功率见 normalize_meta.csv"
```

---

## Stage 3.5：重建样本对

```
请先完整阅读 ROADMAP.md，然后实现以下内容。前提：Stage 3 已完成。

【脚本】src/stage1_data/rebuild_pairs.py
读取 outputs/iris_normalized/normalize_meta.csv，取 status=success 的所有 img_id，
这就是全量可用图片集合（比 Stage 1 的初始集合大得多）。
读取 data/datasetXGN/relations.csv，注意该文件无 header 行，必须用：
  rel = pd.read_csv('data/datasetXGN/relations.csv', header=None, names=['blood_id', 'img_id'])
  rel['img_id'] = rel['img_id'].astype(str)
不要使用 blood.csv（已弃用）。
将 relations.csv 过滤到全量可用集合：rel = rel[rel['img_id'].isin(success_imgs)]
正样本对：同一 blood_id 下，在可用集合中存在的图片两两组合（itertools.combinations），label=1。
  去重：pairs_pos = list(set(frozenset(p) for p in pairs_pos))，再转回 (a, b) 元组。
负样本对：跨 blood_id 随机采样，数量 = 正样本数 × 2，seed=42，label=0。
按 8:2 分割，覆盖写入 data/pairs_train.csv 和 data/pairs_val.csv。
运行完打印：新的正样本对数、负样本对数，以及与 Stage 1 版本相比增加了多少。
完成后执行：git add src/stage1_data/rebuild_pairs.py data/pairs_train.csv data/pairs_val.csv && git commit -m "feat(stage3.5): 全量样本对重建，正样本对扩充"
```

---

## Stage 4：孪生网络训练

```
请先完整阅读 ROADMAP.md，然后实现以下内容。前提：Stage 3.5 已完成，data/pairs_train.csv 为最新版本。

【文件一】configs/siamese.yaml
按 ROADMAP.md 中的配置创建此文件。

【文件二】src/stage4_siamese/model.py
实现 IrisEncoder(feat_dim=128)：
使用 MobileNetV2 预训练 backbone（去掉分类头），接 AdaptiveAvgPool2d(1)，
flatten 后接 Linear(1280, feat_dim)，输出做 F.normalize(x, p=2, dim=1)。
实现 SiameseNet(encoder)：forward(img_a, img_b) 返回 (feat_a, feat_b)。

【文件三】src/stage4_siamese/dataset.py
实现 PairDataset(pairs_csv, img_dir, transform=None)。
读取 pairs_csv（列：img_id_a, img_id_b, label）。
加载 img_dir/{img_id}.png，转为 RGB，resize 到 128×128。
transform 默认：ToTensor + Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])。
跳过任意一张图片文件不存在的样本对（打印 warning，不报错）。

【文件四】src/stage4_siamese/loss.py
实现 contrastive_loss(feat_a, feat_b, label, margin=1.0)。
公式：L = label * d^2 + (1-label) * clamp(margin-d, 0)^2，返回 batch 均值。
d = torch.norm(feat_a - feat_b, dim=1)。

【文件五】src/stage4_siamese/train.py
读取 configs/siamese.yaml 配置。
Adam 优化器 + CosineAnnealingLR 调度器。
每个 epoch 记录 train_loss 和 val_loss，写入 logs/siamese_train.log，同时写 TensorBoard。
val_loss 最优时保存到 checkpoints/siamese/best.pt。
每 10 个 epoch 在验证集上计算 Recall@1：
  对验证集中所有 label=1 的对，计算 anchor 图片特征在全部验证集特征中的最近邻，
  若最近邻正好是正样本图，则命中，统计命中率。
支持 --resume 从上次中断的 checkpoint 继续训练。

【文件六】src/stage4_siamese/build_db.py
加载 checkpoints/siamese/best.pt 中的 IrisEncoder，设置为 eval 模式。
读取 outputs/iris_normalized/normalize_meta.csv，得到 status=success 的 img_id 集合。
读取 data/datasetXGN/pigeon.csv，用 ID 列对应图片 ID（读取时将 ID 列强制转为 str 类型），筛选 BLOOD 字段非空且该 BLOOD 样本数 ≥ 50 的图片。
取两者交集，对每张图提取 128-dim 特征向量。
保存 outputs/features/feature_db.npy（shape: N×128，float32）。
保存 outputs/features/feature_db_meta.csv（列：img_id, pg_id, blood）。
构建 faiss.IndexFlatL2(128)，add 全部向量，序列化到 outputs/features/faiss_index.bin。
运行完打印：入库图片数、覆盖品系数。

运行 train.py，确认训练正常启动并打印第一个 epoch 的 loss。
训练完成后运行 build_db.py。
完成后执行：git add src/stage4_siamese/ configs/siamese.yaml outputs/features/feature_db_meta.csv && git commit -m "feat(stage4): 孪生网络训练脚本和特征库构建脚本"
```

---

## Stage 5：后端服务

```
请先完整阅读 ROADMAP.md，然后实现以下内容。前提：Stage 4 已完成。

【文件一】src/stage5_server/pipeline.py
实现类 IrisPipeline：
__init__：加载 IrisEncoder（checkpoints/siamese/best.pt），加载 FAISS 索引（outputs/features/faiss_index.bin），读取 feature_db_meta.csv。
方法 encode(img_bytes) -> 128-dim numpy 向量，按 ROADMAP.md 的三种情况处理输入图片：
  - 宽高比 > 4：视为已归一化的纹理图，直接 resize 到 128×128
  - 宽高比 0.5~2：视为眼部特写，走 iris_localize + normalize_iris
  - 其他：视为原始全图，先走 YOLO 检测找眼部区域，再走上一步
  任意步骤失败抛出带中文说明的 ValueError。
方法 compare(img_bytes_a, img_bytes_b) -> dict{distance: float, result: str, threshold: float}。
  distance 为 L2 欧氏距离，result 为"可能是同一家族"或"品种差异较大"。
方法 search(img_bytes, top_k=4) -> list of dict{rank, img_id, pg_id, blood, distance}。

【文件二】src/stage5_server/app.py
Flask 应用，启动时初始化 IrisPipeline（只初始化一次）。
POST /compare：接收 multipart/form-data 的 image_a 和 image_b，返回 JSON。
POST /search：接收 multipart/form-data 的 image 和可选 top_k，返回 JSON。
GET /：返回 templates/index.html。
所有异常捕获后返回 HTTP 400 和中文错误信息，不能让服务崩溃。
支持命令行参数 --host（默认0.0.0.0）、--port（默认5000）、--threshold（默认1.0）。

【文件三】src/stage5_server/threshold.py
加载 IrisPipeline。
读取 data/pairs_val.csv，批量计算所有验证对的 L2 距离。
计算 ROC 曲线，找出 FPR + FNR 最小的阈值点。
打印推荐阈值和对应的 FPR/FNR 数值。
可选：将 ROC 曲线保存为 outputs/roc_curve.png。

【文件四】src/stage5_server/templates/index.html
简洁的 Web 演示页，两个 Tab："品种比对" 和 "品种检索"。
品种比对 Tab：两个图片上传框 + 比对按钮 + 结果区域（显示距离数值和判断结论）。
品种检索 Tab：一个图片上传框 + 检索按钮 + 4宫格结果展示（每格显示缩略图、环号、品系名）。
使用原生 JS fetch API，不引入任何外部框架。上传后在结果区显示 loading 状态。

【文件五】src/stage5_server/requirements.txt

启动服务后，用 curl 分别测试 /compare 和 /search 接口，确认返回正确 JSON 格式。
完成后执行：git add src/stage5_server/ && git commit -m "feat(stage5): Flask 后端服务，比对和检索接口"
```

---

## Stage 6：Android 部署

```
请先完整阅读 ROADMAP.md，然后实现以下内容。前提：Stage 4 已完成，checkpoints/siamese/best.pt 存在。

【脚本一】src/stage6_android/export_onnx.py
加载 src/stage4_siamese/model.py 中的 IrisEncoder，加载 checkpoints/siamese/best.pt 权重。
导出 ONNX 到 checkpoints/siamese/encoder.onnx，input shape=(1,3,128,128)，opset_version=11。
用 onnxruntime 验证：对同一随机输入，PyTorch 输出与 ONNX 输出最大误差 < 1e-4。
打印输入/输出 shape 确认。

【文件二】src/stage6_android/jni/iris_encoder.cpp
实现两个 JNI 函数：
init(paramPath, binPath)：ncnn::Net load_param + load_model，返回是否成功。
encode(bitmap)：AndroidBitmap_lockPixels → ncnn::Mat::from_android_bitmap_resize 到 128×128
→ 减均值除标准差（mean=127.5, norm=1/127.5）→ 网络前向推理 → 提取输出层
→ L2 归一化 → 返回 jfloatArray（128 个 float）。
包含必要的 ncnn 和 android jni 头文件引用。

【文件三】src/stage6_android/android_app/ 完整 Android Studio 工程
参考 https://github.com/nihui/ncnn-android-mobilenetssd 的工程结构。
主界面两个 Tab："品种比对" 和 "品种检索"。
品种比对：两个 ImageView + 选择图片按钮 + 比对按钮 + 结果 TextView（显示距离和判断）。
品种检索：一个 ImageView + 选择图片按钮 + 检索按钮，结果用 RecyclerView 展示（调用 Stage 5 的 HTTP 接口）。
IrisEncoder.java 包含 native init() 和 encode() 声明。
CMakeLists.txt 正确链接 ncnn 和 android log 库。

【文件四】src/stage6_android/DEPLOY.md
包含：模型转换完整步骤（PyTorch→ONNX→NCNN→量化float16）、
Android Studio 配置步骤（SDK路径、NDK路径、ncnn版本替换）、常见报错排查。

运行 export_onnx.py，确认 encoder.onnx 成功生成且验证通过。
完成后执行：git add src/stage6_android/ && git commit -m "feat(stage6): Android 部署代码，ONNX导出和NCNN JNI实现"
```

---

## 遇到报错时使用

```
下面这个脚本运行报错了，帮我修复：

文件：[填入报错的脚本路径]

错误信息：
[粘贴完整报错]

请阅读 ROADMAP.md 相关部分和出错的文件，找出原因，修复后重新运行验证。
```

---

## 执行顺序总览

```
Stage 1 → Stage 2（需GPU） → Stage 3 → Stage 3.5 → Stage 4（需GPU） → Stage 5 → Stage 6
```

GPU 不可用时，Stage 2 和 Stage 4 可以先设 epochs=5 跑通流程，确认无误后再正式训练。
