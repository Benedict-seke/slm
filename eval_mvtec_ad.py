"""Compare simple one-class anomaly detectors on MVTec AD."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
REQUIRED_CATEGORY_DIRS = ("train", "test", "ground_truth")
MAX_REFERENCE_IMAGES = 32
CLASSIFIER_MODELS = {
    "mobilenet_v3_small": "MobileNetV3-Small.onnx",
    "mobilenet_v3_large": "MobileNetV3-Large.onnx",
    "efficientnet_lite0": "EfficientNet-Lite0.onnx",
}


def discover_categories(root: Path) -> list[Path]:
    """Find MVTec category folders directly under root."""
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and all((path / name).is_dir() for name in REQUIRED_CATEGORY_DIRS)
        and (path / "train" / "good").is_dir()
        and (path / "test" / "good").is_dir()
    )


def _image_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _read_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _classifier_tensor(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = 256 / min(width, height)
        image = image.resize(
            (max(256, round(width * scale)), max(256, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        width, height = image.size
        left, top = (width - 224) // 2, (height - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return ((array - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)


def _read_mask(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.uint8) > 0


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    positives = int(labels.sum())
    negatives = labels.size - positives
    if not positives or not negatives:
        raise ValueError("ROC AUC requires at least one positive and one negative sample")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    )
    ends = np.r_[starts[1:], scores.size]
    ranks = np.empty(scores.size, dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2

    positive_rank_sum = ranks[labels].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _training_statistics(train_dir: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    images = _image_files(train_dir)
    if not images:
        raise ValueError(f"No training images found in {train_dir}")

    mean = np.zeros((size, size, 3), dtype=np.float64)
    squared_deviations = np.zeros_like(mean)
    for count, path in enumerate(images, start=1):
        image = _read_rgb(path, size).astype(np.float64)
        delta = image - mean
        mean += delta / count
        squared_deviations += delta * (image - mean)

    variance = squared_deviations / max(len(images) - 1, 1)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _texture_features(image: np.ndarray) -> np.ndarray:
    gray = np.mean(image * np.array([0.299, 0.587, 0.114], dtype=np.float32), axis=2)
    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.hypot(grad_x, grad_y)
    laplacian = np.gradient(grad_x, axis=1) + np.gradient(grad_y, axis=0)
    return np.stack((gray, grad_mag, laplacian), axis=2)


def _gaussian_map(
    image: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return np.mean(np.abs(image - mean) / np.maximum(std, 0.03), axis=2)


def _nearest_neighbor_map(image: np.ndarray, references: np.ndarray) -> np.ndarray:
    distances = np.mean(np.square(references - image[None, ...]), axis=3)
    return np.sqrt(np.min(distances, axis=0))


def _standardize_embeddings(
    references: np.ndarray, queries: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = references.mean(axis=0, keepdims=True)
    std = np.maximum(references.std(axis=0, keepdims=True), 1e-5)
    return (references - mean) / std, (queries - mean) / std


def _embedding_knn_scores(
    references: np.ndarray, queries: np.ndarray
) -> np.ndarray:
    normalized_references, normalized_queries = _standardize_embeddings(
        references, queries
    )
    distances = np.sum(
        np.square(normalized_queries[:, None, :] - normalized_references[None, :, :]),
        axis=2,
    )
    return np.sqrt(np.min(distances, axis=1) / references.shape[1])


class OnnxClassifier:
    """Use an ImageNet classifier's logits as a compact one-class descriptor."""

    def __init__(self, name: str, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained classifier ONNX not found: {path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.name = name
        self.path = path
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, path: Path) -> tuple[np.ndarray, float]:
        tensor = _classifier_tensor(path)
        started = time.perf_counter()
        logits = self.session.run(None, {self.input_name: tensor})[0]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return np.asarray(logits, dtype=np.float32).reshape(-1), elapsed_ms


def _find_mask(category: Path, defect_type: str, image_path: Path) -> Path:
    mask_dir = category / "ground_truth" / defect_type
    for stem in (f"{image_path.stem}_mask", image_path.stem):
        for extension in sorted(IMAGE_EXTENSIONS):
            candidate = mask_dir / f"{stem}{extension}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"No ground-truth mask found for {image_path} in {mask_dir}"
    )


def evaluate_category(
    category: Path,
    size: int,
    classifiers: dict[str, OnnxClassifier],
) -> list[dict[str, float | int | str]]:
    mean, std = _training_statistics(category / "train" / "good", size)
    texture_train_dir = category / "train" / "good"
    texture_images = [
        _texture_features(_read_rgb(path, size))
        for path in _image_files(texture_train_dir)
    ]
    texture_stack = np.stack(texture_images)
    texture_mean = texture_stack.mean(axis=0)
    texture_std = np.maximum(texture_stack.std(axis=0, ddof=1), 0.03)
    train_images = _image_files(texture_train_dir)
    selected = np.linspace(
        0, len(train_images) - 1, min(MAX_REFERENCE_IMAGES, len(train_images)), dtype=int
    )
    references = np.stack([_read_rgb(train_images[index], size) for index in selected])

    method_names = (
        "rgb_gaussian",
        "rgb_spatial_knn",
        "texture_gaussian",
        *classifiers.keys(),
    )
    image_scores: dict[str, list[float]] = {name: [] for name in method_names}
    image_labels: list[bool] = []
    pixel_scores: dict[str, list[np.ndarray]] = {name: [] for name in method_names}
    pixel_labels: list[np.ndarray] = []
    classifier_train_embeddings: dict[str, np.ndarray] = {}
    classifier_ms: dict[str, float] = {}
    if classifiers:
        selected_paths = [train_images[index] for index in selected]
        for name, classifier in classifiers.items():
            embedded = [classifier.embed(path) for path in selected_paths]
            classifier_train_embeddings[name] = np.stack(
                [item[0] for item in embedded]
            )
            classifier_ms[name] = sum(item[1] for item in embedded)

    test_dir = category / "test"
    test_embeddings: dict[str, list[np.ndarray]] = {
        name: [] for name in classifiers
    }
    for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        is_anomaly = defect_dir.name != "good"
        for image_path in _image_files(defect_dir):
            image = _read_rgb(image_path, size)
            maps = {
                "rgb_gaussian": _gaussian_map(image, mean, std),
                "rgb_spatial_knn": _nearest_neighbor_map(image, references),
                "texture_gaussian": _gaussian_map(
                    _texture_features(image), texture_mean, texture_std
                ),
            }
            for name, classifier in classifiers.items():
                embedding, elapsed_ms = classifier.embed(image_path)
                classifier_ms[name] += elapsed_ms
                test_embeddings[name].append(embedding)
            for method, anomaly_map in maps.items():
                top_count = max(1, int(anomaly_map.size * 0.01))
                image_scores[method].append(
                    float(
                        np.partition(anomaly_map.ravel(), -top_count)[-top_count:].mean()
                    )
                )
                pixel_scores[method].append(anomaly_map.ravel())
            image_labels.append(is_anomaly)

            if is_anomaly:
                mask_path = _find_mask(category, defect_dir.name, image_path)
                mask = _read_mask(mask_path, size)
            else:
                mask = np.zeros((size, size), dtype=bool)
            pixel_labels.append(mask.ravel())

    if not image_labels:
        raise ValueError(f"No test images found in {test_dir}")
    labels = np.array(image_labels)
    pixel_ground_truth = np.concatenate(pixel_labels)

    results: list[dict[str, float | int | str]] = []
    for method in method_names:
        if method in classifiers:
            test_vectors = np.stack(test_embeddings[method])
            scores = _embedding_knn_scores(
                classifier_train_embeddings[method], test_vectors
            )
            results.append(
                {
                    "model": method,
                    "category": category.name,
                    "train_good": len(train_images),
                    "train_references": len(references),
                    "test_images": len(image_labels),
                    "image_auroc": _roc_auc(scores, labels),
                    "pixel_auroc": "",
                    "model_size_mib": round(
                        classifiers[method].path.stat().st_size / (1024 * 1024), 2
                    ),
                    "inference_ms_per_image": round(
                        classifier_ms[method] / (len(selected) + len(image_labels)), 2
                    ),
                }
            )
            continue
        results.append(
            {
                "model": method,
                "category": category.name,
                "train_good": len(train_images),
                "train_references": len(references),
                "test_images": len(image_labels),
                "image_auroc": _roc_auc(np.array(image_scores[method]), labels),
                "pixel_auroc": _roc_auc(
                    np.concatenate(pixel_scores[method]), pixel_ground_truth
                ),
                "model_size_mib": "",
                "inference_ms_per_image": "",
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare lightweight ImageNet CNN descriptors and one-class baselines on MVTec AD."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Directory containing category folders (default: data)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Optional category names, e.g. bottle metal_nut",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Square evaluation resolution in pixels (default: 128)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_cv/mvtec_ad.csv"),
        help="CSV output path (default: results_cv/mvtec_ad.csv)",
    )
    parser.add_argument(
        "--classifiers",
        nargs="*",
        choices=tuple(CLASSIFIER_MODELS),
        default=list(CLASSIFIER_MODELS),
        help="Lightweight ImageNet ONNX classifiers to include (default: all three)",
    )
    args = parser.parse_args()
    if args.size < 16:
        parser.error("--size must be at least 16")

    categories = discover_categories(args.root)
    if args.categories:
        by_name = {path.name: path for path in categories}
        missing = [name for name in args.categories if name not in by_name]
        if missing:
            parser.error(
                f"Categories not found under {args.root}: {', '.join(missing)}"
            )
        categories = [by_name[name] for name in args.categories]
    if not categories:
        parser.error(
            f"No MVTec category folders found under {args.root}; "
            "expected train/good, test/good, and ground_truth directories."
        )

    classifiers = {
        name: OnnxClassifier(name, Path("models/cv") / CLASSIFIER_MODELS[name])
        for name in args.classifiers
    }
    results: list[dict[str, float | int | str]] = []
    for category in categories:
        category_results = evaluate_category(category, args.size, classifiers)
        results.extend(category_results)
        for result in category_results:
            print(
                f"{result['category']:<14} {result['model']:<18} "
                f"test={result['test_images']:>3} "
                f"image AUROC={result['image_auroc']:.4f} "
                f"pixel AUROC={result['pixel_auroc'] or 'n/a'}"
                + (
                    f" {result['model_size_mib']} MiB, "
                    f"{result['inference_ms_per_image']} ms/img"
                    if result["model_size_mib"]
                    else ""
                )
            )

    method_names = sorted({str(row["model"]) for row in results})
    for method in method_names:
        method_results = [row for row in results if row["model"] == method]
        results.append(
            {
                "model": method,
                "category": "MACRO_AVERAGE",
                "train_good": "",
                "train_references": MAX_REFERENCE_IMAGES,
                "test_images": sum(int(row["test_images"]) for row in method_results),
                "image_auroc": float(
                    np.mean([row["image_auroc"] for row in method_results])
                ),
                "pixel_auroc": (
                    float(
                        np.mean(
                            [
                                float(row["pixel_auroc"])
                                for row in method_results
                                if row["pixel_auroc"] != ""
                            ]
                        )
                    )
                    if method_results[0]["pixel_auroc"] != ""
                    else ""
                ),
                "model_size_mib": method_results[0]["model_size_mib"],
                "inference_ms_per_image": (
                    round(
                        float(
                            np.mean(
                                [
                                    float(row["inference_ms_per_image"])
                                    for row in method_results
                                ]
                            )
                        ),
                        2,
                    )
                    if method_results[0]["inference_ms_per_image"] != ""
                    else ""
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    for result in results[-len(method_names):]:
        print(
            f"{result['category']:<14} {result['model']:<18} "
            f"image AUROC={result['image_auroc']:.4f} "
            f"pixel AUROC={result['pixel_auroc'] or 'n/a'}"
            + (
                f" {result['model_size_mib']} MiB, "
                f"{result['inference_ms_per_image']} ms/img"
                if result["model_size_mib"]
                else ""
            )
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
