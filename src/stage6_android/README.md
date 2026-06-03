# PigeonVision Android 客户端使用说明

Android 端负责拍照或选图、用 NCNN YOLOv5s 检测并裁剪鸽眼，然后把眼部裁剪图提交给 Flask 服务端。服务端继续完成 U-Net 虹膜分割、特征提取、FAISS 检索和比对。

## 前置条件

- Android Studio 或本目录 Gradle Wrapper 可用
- Android SDK / NDK / CMake 已安装
- 真机与服务端电脑在同一局域网
- 服务端已启动：

```bash
cd /path/to/PigeonVision
python src/stage5_server/app.py --host 0.0.0.0 --port 5000
```

- NCNN YOLO 模型文件存在：

```text
app/src/main/assets/yolo/model.ncnn.param
app/src/main/assets/yolo/model.ncnn.bin
```

## 构建 APK

```bash
cd src/stage6_android
./gradlew :app:assembleDebug
```

生成文件：

```text
app/build/outputs/apk/debug/app-debug.apk
```

## App 配置

1. 打开 App 后，在顶部服务器地址输入框填写 Flask 服务地址。
2. 真机演示时不要填 `localhost`，应填写电脑局域网 IP，例如：

```text
http://192.168.1.23:5000
```

3. 点击「连接测试」。成功时会显示图库规模。

服务端地址会写入 `SharedPreferences`，下次启动自动恢复。

## 品种比对

1. 进入「品种比对」页。
2. 选择或拍摄两张鸽子图。
3. 点击「开始比对」。

流程：

```text
原图 -> Android NCNN YOLO 检测眼部 -> JPEG 眼部裁剪图 -> POST /compare eye_crop=1
```

结果展示：

- 欧氏距离
- 判定阈值
- 是否同一品种

## 品种检索

1. 进入「品种检索」页。
2. 选择或拍摄一张鸽子图。
3. 输入 Top-K，范围为 `1-100`，默认 `20`。
4. 点击「开始检索」。

流程：

```text
原图 -> Android NCNN YOLO 检测眼部 -> JPEG 眼部裁剪图 -> POST /search eye_crop=1 top_k=N
```

结果展示：

- 每页 10 条，超过 10 条自动分页
- 排名、品系名、PG_ID、blood_id、img_id、距离
- 服务端 `/image/<img_id>` 返回的原始鸽眼图缩略图

## 服务端接口依赖

### GET /health

用于连接测试。

```json
{"status": "ok", "gallery_size": 22043, "breed_count": 1234}
```

### POST /compare

```text
multipart/form-data:
  image_a: <JPEG bytes>
  image_b: <JPEG bytes>
  eye_crop: "1"
```

### POST /search

```text
multipart/form-data:
  image: <JPEG bytes>
  top_k: "20"
  eye_crop: "1"
```

返回中的 `image_url` 用于加载缩略图：

```json
{
  "results": [
    {
      "rank": 1,
      "img_id": "571835",
      "blood_id": "B01-123",
      "blood_name": "桑杰士",
      "pg_id": "2016-26-0571835",
      "distance": 0.21,
      "image_url": "/image/571835"
    }
  ]
}
```

### GET /image/<img_id>

返回 `outputs/img_index.csv` 中对应的原始鸽眼图。服务端必须能访问 `data/extracted/{1..12}/` 原始图片。

## 常见问题

- 「连接失败」：确认服务端用 `--host 0.0.0.0` 启动，手机和电脑在同一网络，防火墙放行 5000 端口。
- 「未检测到眼部，请靠近拍摄」：拍摄时让鸽眼占画面更大，避免反光和模糊。
- 检索结果无缩略图：确认服务端存在 `outputs/img_index.csv`，且本机能访问 CSV 中记录的原始图片路径。
- Debug 构建能访问 HTTP 明文地址，正式发布如需 HTTPS 可再收紧网络配置。
