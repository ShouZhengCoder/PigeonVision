"""
模拟 Android 端调用的本地测试脚本。
用法（在项目根目录执行）：
    python src/stage5_server/test_android_sim.py [--mode pipeline|http] [--host 127.0.0.1:8080]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _pick_test_image() -> Path:
    """从 eye_crops 目录取第一张图作为测试图片。"""
    crop_dir = ROOT / "outputs" / "eye_crops"
    images = sorted(crop_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"eye_crops 目录为空：{crop_dir}")
    return images[0]


def test_pipeline_direct(eye_crop_path: Path) -> None:
    """直接调用 pipeline，不启动 Flask，测试 eye_crop=True 分支。"""
    sys.path.insert(0, str(ROOT / "src" / "stage5_server"))
    from pipeline import IrisPipeline  # noqa: E402

    print(f"[pipeline] 加载模型中（首次较慢）...")
    t0 = time.time()
    pipe = IrisPipeline()
    print(f"[pipeline] 加载完成，耗时 {time.time() - t0:.1f}s")
    print(f"[pipeline] gallery_size={pipe.gallery_size}, breed_count={pipe.breed_count}")
    print(f"[pipeline] threshold={pipe.threshold:.4f}")

    img_bytes = eye_crop_path.read_bytes()
    print(f"\n[test] 测试图片：{eye_crop_path.name}（{len(img_bytes)/1024:.1f} KB）")

    # --- search with eye_crop=True ---
    print("\n[test] /search  eye_crop=True ...")
    t0 = time.time()
    results = pipe.search(img_bytes, top_k=5, eye_crop=True)
    elapsed = time.time() - t0
    print(f"  耗时 {elapsed*1000:.0f}ms，返回 {len(results)} 条结果：")
    for r in results:
        print(f"  rank={r['rank']}  blood={r['blood_name']!r}  dist={r['distance']:.4f}  pg_id={r.get('pg_id','')}")

    # --- compare with eye_crop=True (same image vs same image) ---
    print("\n[test] /compare  eye_crop=True（同图自比，距离应接近 0）...")
    t0 = time.time()
    cmp = pipe.compare(img_bytes, img_bytes, eye_crop=True)
    elapsed = time.time() - t0
    print(f"  耗时 {elapsed*1000:.0f}ms")
    print(f"  distance={cmp['distance']:.6f}  same_family={cmp['same_family']}  threshold={cmp['threshold']:.4f}")
    if cmp["distance"] > 0.1:
        print("  ⚠️  同图自比距离 > 0.1，embedding 可能有问题")
    else:
        print("  ✅ 同图自比通过")

    # --- search with eye_crop=False（走 YOLO 路径，对比耗时）---
    print("\n[test] /search  eye_crop=False（走 YOLO，对比耗时）...")
    t0 = time.time()
    try:
        results2 = pipe.search(img_bytes, top_k=5, eye_crop=False)
        elapsed2 = time.time() - t0
        print(f"  耗时 {elapsed2*1000:.0f}ms，返回 {len(results2)} 条")
        if results and results2:
            same = results[0]["img_id"] == results2[0]["img_id"]
            print(f"  Top-1 一致：{'✅' if same else '⚠️ 不一致'} "
                  f"(eye_crop={results[0]['img_id']} vs yolo={results2[0]['img_id']})")
    except Exception as e:
        elapsed2 = time.time() - t0
        print(f"  ⚠️  YOLO 路径失败（{elapsed2*1000:.0f}ms）：{e}")
        print("  → 这正是 eye_crop=True 必须的原因")


def test_http(eye_crop_path: Path, host: str) -> None:
    """通过 HTTP 调用运行中的 Flask 服务，模拟 Android POST 请求。"""
    import urllib.request
    import urllib.error
    import json

    base = f"http://{host}"

    # health check
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
            health = json.loads(resp.read())
        print(f"[http] /health OK: {health}")
    except Exception as e:
        print(f"[http] 连接失败：{e}")
        print("  请先启动服务：")
        print("    conda activate pigeonvision")
        print("    python src/stage5_server/app.py --host 0.0.0.0 --port 8080")
        return

    img_bytes = eye_crop_path.read_bytes()
    print(f"\n[test] 测试图片：{eye_crop_path.name}（{len(img_bytes)/1024:.1f} KB）")

    def multipart_post(url, fields, files):
        """最小化 multipart/form-data 实现，无需 requests 库。"""
        boundary = b"----PigeonVisionTestBoundary"
        body = b""
        for k, v in fields.items():
            body += b"--" + boundary + b"\r\n"
            body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
            body += str(v).encode() + b"\r\n"
        for k, (fname, fdata) in files.items():
            body += b"--" + boundary + b"\r\n"
            body += f'Content-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'.encode()
            body += b"Content-Type: image/jpeg\r\n\r\n"
            body += fdata + b"\r\n"
        body += b"--" + boundary + b"--\r\n"
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    # search eye_crop=1
    print("\n[test] POST /search  eye_crop=1 ...")
    t0 = time.time()
    result = multipart_post(
        f"{base}/search",
        fields={"top_k": "5", "eye_crop": "1"},
        files={"image": (eye_crop_path.name, img_bytes)},
    )
    elapsed = time.time() - t0
    print(f"  耗时 {elapsed*1000:.0f}ms")
    for r in result.get("results", []):
        print(f"  rank={r['rank']}  blood={r['blood_name']!r}  dist={r['distance']:.4f}")

    # compare same image, eye_crop=1
    print("\n[test] POST /compare  eye_crop=1（同图自比）...")
    t0 = time.time()
    cmp = multipart_post(
        f"{base}/compare",
        fields={"eye_crop": "1"},
        files={
            "image_a": (eye_crop_path.name, img_bytes),
            "image_b": (eye_crop_path.name, img_bytes),
        },
    )
    elapsed = time.time() - t0
    print(f"  耗时 {elapsed*1000:.0f}ms")
    print(f"  distance={cmp['distance']:.6f}  same_family={cmp['same_family']}  threshold={cmp['threshold']:.4f}")
    if cmp["distance"] > 0.1:
        print("  ⚠️  同图自比距离 > 0.1")
    else:
        print("  ✅ 同图自比通过")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pipeline", "http"], default="pipeline",
                        help="pipeline: 直接调用（无需启动服务）；http: 通过 HTTP 测试运行中的服务")
    parser.add_argument("--host", default="127.0.0.1:8080", help="Flask 地址（仅 http 模式）")
    parser.add_argument("--image", default=None, help="指定测试图片路径（默认自动选取）")
    args = parser.parse_args()

    img_path = Path(args.image) if args.image else _pick_test_image()
    print(f"[config] mode={args.mode}  image={img_path}")

    if args.mode == "pipeline":
        test_pipeline_direct(img_path)
    else:
        test_http(img_path, args.host)


if __name__ == "__main__":
    main()
