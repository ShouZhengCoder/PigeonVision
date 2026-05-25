from __future__ import annotations

import argparse
import sys
import traceback

from flask import Flask, jsonify, render_template, request

try:
    from .pipeline import IrisPipeline
except ImportError:
    from pipeline import IrisPipeline


app = Flask(__name__, template_folder="templates")
pipeline = IrisPipeline()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5 Flask service for iris compare and search.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


@app.errorhandler(Exception)
def handle_error(exc: Exception):
    traceback.print_exc(file=sys.stderr)
    return jsonify({"error": f"处理失败：{exc}"}), 400


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "gallery_size": pipeline.gallery_size,
            "breed_count": pipeline.breed_count,
        }
    )


@app.post("/compare")
def compare():
    image_a = request.files.get("image_a")
    image_b = request.files.get("image_b")
    if image_a is None or image_b is None:
        raise ValueError("请上传 image_a 和 image_b")
    result = pipeline.compare(image_a.read(), image_b.read())
    return jsonify(result)


@app.post("/search")
def search():
    return _search_from_request()


@app.post("/search_raw")
def search_raw():
    return _search_from_request()


def _search_from_request():
    image = request.files.get("image")
    if image is None:
        raise ValueError("请上传 image")
    raw_top_k = request.form.get("top_k", "10")
    try:
        top_k = int(raw_top_k)
    except ValueError as exc:
        raise ValueError("top_k 必须是整数") from exc
    results = pipeline.search(image.read(), top_k=top_k)
    return jsonify({"results": results})


if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port)
