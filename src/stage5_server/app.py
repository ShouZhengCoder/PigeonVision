from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import threading
import traceback
from io import BytesIO

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(thread_env, "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
faulthandler.enable(all_threads=True)

from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from werkzeug.exceptions import HTTPException

try:
    from .pipeline import DEFAULT_SEARCH_TOP_K, MAX_SEARCH_TOP_K, IrisPipeline
except ImportError:
    from pipeline import DEFAULT_SEARCH_TOP_K, MAX_SEARCH_TOP_K, IrisPipeline


app = Flask(__name__, template_folder="templates")
pipeline = IrisPipeline()
pipeline_lock = threading.RLock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5 Flask service for iris compare and search.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Server host (use 0.0.0.0 to expose on all interfaces)")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


@app.errorhandler(Exception)
def handle_error(exc: Exception):
    if isinstance(exc, HTTPException):
        return exc
    traceback.print_exc(file=sys.stderr)
    return jsonify({"error": f"处理失败：{exc}"}), 400


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    with pipeline_lock:
        gallery_size = pipeline.gallery_size
        breed_count = pipeline.breed_count
    return jsonify(
        {
            "status": "ok",
            "gallery_size": gallery_size,
            "breed_count": breed_count,
        }
    )


@app.post("/compare")
def compare():
    image_a = request.files.get("image_a")
    image_b = request.files.get("image_b")
    if image_a is None or image_b is None:
        raise ValueError("请上传 image_a 和 image_b")
    eye_crop = request.form.get("eye_crop", "0") == "1"
    image_a_bytes = image_a.read()
    image_b_bytes = image_b.read()
    if not eye_crop and not _server_yolo_enabled():
        raise ValueError("服务端 YOLO 原图兜底已禁用，请上传 Android 眼部裁剪图并设置 eye_crop=1")
    with pipeline_lock:
        result = pipeline.compare(image_a_bytes, image_b_bytes, eye_crop=eye_crop)
    return jsonify(result)


@app.post("/search")
def search():
    return _search_from_request()


@app.post("/search_raw")
def search_raw():
    return _search_from_request()


@app.get("/image/<img_id>")
def gallery_image(img_id: str):
    with pipeline_lock:
        image_bytes = pipeline.gallery_image_jpeg(img_id)
    if image_bytes is None:
        abort(404)
    return send_file(
        BytesIO(image_bytes),
        mimetype="image/jpeg",
        download_name=f"{img_id}.jpg",
    )


def _search_from_request():
    image = request.files.get("image")
    if image is None:
        raise ValueError("请上传 image")
    raw_top_k = request.form.get("top_k", str(DEFAULT_SEARCH_TOP_K))
    try:
        top_k = int(raw_top_k)
    except ValueError as exc:
        raise ValueError("top_k 必须是整数") from exc
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    top_k = min(top_k, MAX_SEARCH_TOP_K)
    eye_crop = request.form.get("eye_crop", "0") == "1"
    image_bytes = image.read()
    if not eye_crop and not _server_yolo_enabled():
        raise ValueError("服务端 YOLO 原图兜底已禁用，请上传 Android 眼部裁剪图并设置 eye_crop=1")
    with pipeline_lock:
        response = pipeline.search(image_bytes, top_k=top_k, eye_crop=eye_crop)
    for item in response["results"]:
        img_id = str(item.get("img_id", ""))
        if img_id:
            item["image_url"] = url_for("gallery_image", img_id=img_id)
    return jsonify(response)


def _is_segmentation_failure(exc: Exception) -> bool:
    message = str(exc)
    return "虹膜分割失败" in message or "虹膜展开失败" in message or "ellipse" in message


def _server_yolo_enabled() -> bool:
    return os.environ.get("PIGEONVISION_ENABLE_SERVER_YOLO", "").strip().lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port, threaded=False, use_reloader=False)
