"""Run nnUNetv2 inference and visualize prediction vs ground truth."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATASET_ID = 501
NNUNET_ROOT = Path(__file__).resolve().parents[2] / "nnunet_data"
DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "Common Carotid Artery Ultrasound Images"
    / "US images"
    / "202201121748100022VAS_slice_1176.png"
)


def parse_args():
    parser = argparse.ArgumentParser(description="nnUNetv2 carotid inference")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--fold", default="0", help="Trained fold to use (0-4 or 'all')")
    parser.add_argument("--configuration", "-c", default="2d")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth",
                        choices=["checkpoint_best.pth", "checkpoint_latest.pth", "checkpoint_final.pth"])
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    return parser.parse_args()


def setup_env() -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_ROOT / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(NNUNET_ROOT / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(NNUNET_ROOT / "nnUNet_results")
    return env


def prepare_input(image: Path, input_dir: Path) -> str:
    input_dir.mkdir(parents=True, exist_ok=True)
    for f in input_dir.iterdir():
        f.unlink()
    case_id = image.stem
    dest = input_dir / f"{case_id}_0000.png"
    shutil.copy2(image, dest)
    return case_id


def run_predict(input_dir: Path, output_dir: Path, fold: str, config: str,
                checkpoint: str, env: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in output_dir.iterdir():
        f.unlink()

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(DATASET_ID),
        "-c", config,
        "-f", fold,
        "-chk", checkpoint,
        "--verbose",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def dice_score(pred: np.ndarray, label: np.ndarray) -> float:
    pred = pred > 0
    label = label > 0
    intersection = (pred & label).sum()
    return (2.0 * intersection) / (pred.sum() + label.sum() + 1e-8)


def plot_result(image_path: Path, pred_path: Path, label_path: Path | None) -> float | None:
    image = np.array(Image.open(image_path))
    pred = np.array(Image.open(pred_path))
    if pred.ndim == 3:
        pred = pred[..., 0]

    dice = None
    fig, axes = plt.subplots(1, 3 if label_path else 2, figsize=(14, 5))

    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(pred, cmap="YlOrRd", alpha=0.5)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    if label_path and label_path.exists():
        label = np.array(Image.open(label_path))
        if label.ndim == 3:
            label = label[..., 0]
        label = (label > 0).astype(np.uint8)
        dice = dice_score(pred, label)
        axes[2].imshow(image, cmap="gray")
        axes[2].imshow(label, cmap="Blues", alpha=0.4)
        axes[2].imshow(pred, cmap="YlOrRd", alpha=0.4)
        axes[2].set_title(f"Pred (red) vs GT (blue)\nDice: {dice:.4f}")
        axes[2].axis("off")

    plt.suptitle("nnUNetv2 Carotid Segmentation")
    plt.tight_layout()
    out_plot = NNUNET_ROOT / "inference_output" / "prediction_plot.png"
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_plot}")
    plt.show()
    return dice


def main():
    args = parse_args()
    env = setup_env()
    image = args.image.resolve()

    if not image.exists():
        sys.exit(f"Image not found: {image}")

    input_dir = NNUNET_ROOT / "inference_input"
    output_dir = NNUNET_ROOT / "inference_output"
    case_id = prepare_input(image, input_dir)

    run_predict(input_dir, output_dir, args.fold, args.configuration,
                args.checkpoint, env)

    pred_path = output_dir / f"{case_id}.png"
    if not pred_path.exists():
        preds = list(output_dir.glob("*.png"))
        pred_path = preds[0] if preds else None
    if pred_path is None:
        sys.exit(f"No prediction found in {output_dir}")

    print(f"Prediction saved to {pred_path}")

    label_path = image.parent.parent / "Expert mask images" / image.name
    if not args.no_plot:
        dice = plot_result(image, pred_path, label_path)
        if dice is not None:
            print(f"Dice score vs ground truth: {dice:.4f}")


if __name__ == "__main__":
    main()
