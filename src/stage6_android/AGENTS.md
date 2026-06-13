# PigeonVision Android App — Codex 任务说明

## 项目背景

这是「基于虹膜图像分析的信鸽品种识别系统」的 Android 客户端。

整体架构：
- **Android 端**（本项目）：拍照 → NCNN 运行 YOLOv5s 检测眼部 → 裁剪眼部图 → POST 给服务端
- **服务端**（Flask，已完成）：接收眼部裁剪图 → U-Net 虹膜分割 → 特征提取 → FAISS 检索 → 返回结果

Android 端**不做**虹膜分割和特征提取，只负责眼部检测和 HTTP 通信。

---

## 需要实现的功能

### 功能一：品种比对（Compare）
1. 用户选择或拍摄两张鸽子图片（image_a、image_b）
2. 对每张图分别运行 YOLO 检测，裁剪出眼部区域
3. POST 到服务端 `/compare`，附带参数 `eye_crop=1`
4. 展示结果：欧氏距离、是否同一品种（same_family）

### 功能二：品种检索（Search）
1. 用户选择或拍摄一张鸽子图片
2. YOLO 检测，裁剪眼部
3. POST 到服务端 `/search`，附带参数 `eye_crop=1`、`top_k=<用户输入>`
4. 展示检索结果列表：排名、品系名、PG_ID、blood_id、img_id、距离和服务端图片缩略图
5. Top-K 默认 20，可输入 1-100；结果每页 10 条分页展示

---

## 服务端接口

服务端运行在同一局域网电脑上，地址由用户在 App 内输入（默认 `http://192.168.1.x:8080`）。

### POST /compare
```
multipart/form-data:
  image_a: <JPEG bytes>   # 眼部裁剪图
  image_b: <JPEG bytes>   # 眼部裁剪图
  eye_crop: "1"
```
返回：
```json
{"distance": 0.83, "same_family": true, "threshold": 2.348}
```

### POST /search
```
multipart/form-data:
  image: <JPEG bytes>     # 眼部裁剪图
  top_k: "20"
  eye_crop: "1"
```
返回：
```json
{"results": [{"rank":1, "img_id":"571835", "blood_id":"B01-123", "blood_name":"桑杰士", "distance":0.21, "pg_id":"NL15-1273729", "image_url":"/image/571835"}, ...]}
```

### GET /image/<img_id>
返回 `outputs/img_index.csv` 中对应的原始鸽眼图，Android 端用于检索结果缩略图展示。

### GET /health
返回：`{"status": "ok", "gallery_size": 22043}`

---

## NCNN 模型信息

模型文件位于 `app/src/main/assets/yolo/`：
- `model.ncnn.param`
- `model.ncnn.bin`

模型参数：
- 输入尺寸：416×416，3 通道 RGB
- 输出：检测框，格式 `[batch, 5, 3549]`，其中 5 = [cx, cy, w, h, confidence]（单类别，无需 class score）
- 置信度阈值：0.5（可调）
- NMS IoU 阈值：0.45

### NCNN 推理流程
```
Bitmap (原始图) → resize 到 416×416 → ncnn::Mat::from_pixels_resize
→ 归一化（mean=[0,0,0], norm=[1/255, 1/255, 1/255]）
→ Extractor 推理
→ 解析输出：遍历 3549 个 anchor，过滤 confidence > 0.5
→ NMS
→ 取置信度最高框，按原图比例映射回坐标
→ 扩展 10%（防裁边）后 crop
```

---

## 技术要求

- **语言**：Kotlin
- **UI**：Jetpack Compose（模板默认）
- **NCNN 集成**：使用 Maven 依赖 `com.tencent.ncnn:ncnn-android:1.0.20260526`
- **HTTP**：OkHttp（`com.squareup.okhttp3:okhttp:4.12.0`）
- **图片选择**：系统相册（`ActivityResultContracts.GetContent`）+ 相机（`ActivityResultContracts.TakePicture`）
- **权限**：CAMERA、READ_MEDIA_IMAGES（Android 13+）或 READ_EXTERNAL_STORAGE

---

## UI 结构（简洁即可，课堂演示用）

```
MainActivity（单 Activity）
├── 顶部：服务器地址输入框 + 连接测试按钮（调 /health）
├── Tab 1：Compare
│   ├── 两个图片选择区（点击拍照或选图）
│   ├── 「开始比对」按钮
│   └── 结果展示：距离值 + "同一品种 ✅" 或 "不同品种 ❌"
└── Tab 2：Search
    ├── 一个图片选择区
    ├── Top-K 输入框（默认20，范围1-100）
    ├── 「开始检索」按钮
    └── 结果列表（LazyColumn）：缩略图 | 排名 | 品系名 | PG_ID | 距离，超过10条分页
```

---

## 文件结构目标

```
app/src/main/
├── assets/yolo/
│   ├── model.ncnn.param
│   └── model.ncnn.bin
├── java/com/pigeonvision/app/
│   ├── MainActivity.kt          # 入口，Compose UI
│   ├── YoloDetector.kt          # NCNN 封装，loadModel() + detect()
│   ├── ApiClient.kt             # OkHttp 封装，compare() + search()
│   └── ui/
│       ├── CompareScreen.kt
│       └── SearchScreen.kt
└── cpp/
    └── yolo_jni.cpp             # JNI：Java_com_pigeonvision_app_YoloDetector_*
```

---

## 注意事项

1. NCNN 的 JNI 方法签名必须与 `YoloDetector.kt` 中的 `external fun` 声明一致
2. 模型文件从 assets 读取，用 `context.assets.open(...)` 复制到 `filesDir` 后再加载
3. 网络请求在协程中执行（`Dispatchers.IO`），结果在主线程更新 UI
4. 眼部裁剪图以 JPEG 格式（quality=90）压缩后发送，不发原图
5. 服务器地址持久化到 `SharedPreferences`，App 重启后记住上次填写的地址
6. 如果 YOLO 未检测到眼部（无框超过阈值），Toast 提示用户"未检测到眼部，请靠近拍摄"
