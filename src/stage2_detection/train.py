from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import yaml
from ultralytics import YOLO

from _common import ROOT, ensure_dir, resolve_root_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv5 eye detector with ultralytics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "yolov5.yaml",
        help="Stage 2 training config YAML.",
    )
    parser.add_argument("--data", type=Path, help="YOLO dataset data.yaml path.")
    parser.add_argument("--model", type=str, help="YOLO model name or checkpoint path.")
    parser.add_argument("--epochs", type=int, help="Training epochs.")
    parser.add_argument("--batch", type=int, help="Training batch size.")
    parser.add_argument("--imgsz", type=int, help="Training image size.")
    parser.add_argument("--project", type=Path, help="Checkpoint project directory.")
    parser.add_argument("--name", type=str, help="Run name under project directory.")
    parser.add_argument(
        "--runtime-dataset",
        type=Path,
        help="Runtime YOLO dataset directory with standard images/labels layout.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ultralytics device value. Default lets ultralytics auto-select.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable AMP. Default uses ultralytics behavior.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Optional fraction of the training dataset, useful for smoke tests.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def value(args_value, config: dict[str, object], key: str, default):
    return args_value if args_value is not None else config.get(key, default)


def read_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def symlink_or_replace(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            return
        target.unlink()
    target.symlink_to(source)


def prepare_runtime_dataset(source_data_yaml: Path, runtime_dir: Path) -> Path:
    """Create a YOLO-standard images/labels layout without copying raw images."""
    source_cfg = read_yaml(source_data_yaml)
    source_root = source_data_yaml.parent
    train_txt = resolve_root_path(source_cfg.get("train", source_root / "train.txt"))
    val_txt = resolve_root_path(source_cfg.get("val", source_root / "val.txt"))
    labels_root = source_root / "labels"

    for split, list_path in [("train", train_txt), ("val", val_txt)]:
        if not list_path.exists():
            raise FileNotFoundError(f"Missing {split} image list: {list_path}")
        image_dir = ensure_dir(runtime_dir / "images" / split)
        label_dir = ensure_dir(runtime_dir / "labels" / split)
        with list_path.open("r", encoding="utf-8") as f:
            image_paths = [Path(line.strip()) for line in f if line.strip()]
        for image_path in image_paths:
            if not image_path.exists():
                continue
            label_path = labels_root / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            symlink_or_replace(image_path, image_dir / image_path.name)
            symlink_or_replace(label_path, label_dir / label_path.name)
        print(f"prepared {split}: {len(list(image_dir.glob('*')))} images")

    runtime_yaml = runtime_dir / "data.yaml"
    runtime_yaml.write_text(
        "\n".join(
            [
                f"path: {runtime_dir.resolve()}",
                f"train: {(runtime_dir / 'images' / 'train').resolve()}",
                f"val: {(runtime_dir / 'images' / 'val').resolve()}",
                f"nc: {int(source_cfg.get('nc', 1))}",
                f"names: {source_cfg.get('names', ['eye'])}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return runtime_yaml


def main() -> int:
    args = parse_args()
    config_path = resolve_root_path(args.config)
    config = load_config(config_path)

    source_data = resolve_root_path(value(args.data, config, "data", "data/yolo_dataset/data.yaml"))
    model = str(value(args.model, config, "model", "yolov5s.pt"))
    epochs = int(value(args.epochs, config, "epochs", 100))
    batch = int(value(args.batch, config, "batch", 16))
    imgsz = int(value(args.imgsz, config, "imgsz", 416))
    project = resolve_root_path(value(args.project, config, "project", "checkpoints/detection"))
    name = str(value(args.name, config, "name", "exp"))
    runtime_dataset = resolve_root_path(
        args.runtime_dataset
        or config.get("runtime_dataset", project / "runtime_dataset")
    )

    if not source_data.exists():
        raise FileNotFoundError(f"Missing YOLO data config: {source_data}")
    ensure_dir(project)
    ensure_dir(runtime_dataset)
    data = prepare_runtime_dataset(source_data, runtime_dataset)

    cmd = [
        "yolo",
        "train",
        f"data={data}",
        f"model={model}",
        f"epochs={epochs}",
        f"batch={batch}",
        f"imgsz={imgsz}",
        f"project={project}",
        f"name={name}",
    ]
    if args.device:
        cmd.append(f"device={args.device}")
    if args.amp is not None:
        cmd.append(f"amp={args.amp}")
    if args.fraction is not None:
        cmd.append(f"fraction={args.fraction}")
    print("Command:", " ".join(shlex.quote(str(part)) for part in cmd))

    yolo = YOLO(model)
    train_kwargs = {
        "data": str(data),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "project": str(project),
        "name": name,
    }
    if args.device:
        train_kwargs["device"] = args.device
    if args.amp is not None:
        train_kwargs["amp"] = args.amp
    if args.fraction is not None:
        train_kwargs["fraction"] = args.fraction
    yolo.train(**train_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
