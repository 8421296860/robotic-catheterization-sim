"""Convert the carotid ultrasound dataset to nnUNetv2 format."""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_CarotidArtery"
DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "Common Carotid Artery Ultrasound Images"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare carotid dataset for nnUNetv2")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to 'Common Carotid Artery Ultrasound Images' folder",
    )
    parser.add_argument(
        "--nnunet-raw",
        type=Path,
        default=None,
        help="nnUNet_raw directory (default: ../nnunet_data/nnUNet_raw)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying (masks are always converted and saved)",
    )
    return parser.parse_args()


def get_nnunet_raw(nnunet_raw_arg: Path | None) -> Path:
    if nnunet_raw_arg is not None:
        return nnunet_raw_arg.resolve()
    return (Path(__file__).resolve().parents[2] / "nnunet_data" / "nnUNet_raw").resolve()


def convert_mask(mask_path: Path, output_path: Path) -> None:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(output_path)


def prepare_dataset(source: Path, nnunet_raw: Path, symlink: bool = False) -> Path:
    image_dir = source / "US images"
    mask_dir = source / "Expert mask images"

    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected 'US images' and 'Expert mask images' under {source}"
        )

    dataset_dir = nnunet_raw / DATASET_NAME
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    image_files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() == ".png")
    pairs = []
    for image_path in image_files:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        pairs.append((image_path, mask_path))

    for image_path, mask_path in tqdm(pairs, desc="Converting cases"):
        case_id = image_path.stem
        out_image = images_tr / f"{case_id}_0000.png"
        out_label = labels_tr / f"{case_id}.png"

        if symlink:
            if out_image.exists() or out_image.is_symlink():
                out_image.unlink()
            out_image.symlink_to(image_path.resolve())
        else:
            shutil.copy2(image_path, out_image)

        convert_mask(mask_path, out_label)

    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "carotid_artery": 1},
        "numTraining": len(pairs),
        "file_ending": ".png",
        "name": DATASET_NAME,
        "description": "Common Carotid Artery Ultrasound segmentation dataset",
        "reference": "https://data.mendeley.com/datasets/d4xt63mgjm/1",
        "licence": "See Mendeley dataset license",
        "converted_by": "carotid-segmentation prepare_dataset.py",
    }

    with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Prepared {len(pairs)} cases at {dataset_dir}")
    return dataset_dir


def main():
    args = parse_args()
    nnunet_raw = get_nnunet_raw(args.nnunet_raw)
    nnunet_raw.mkdir(parents=True, exist_ok=True)
    prepare_dataset(args.source.resolve(), nnunet_raw, symlink=args.symlink)


if __name__ == "__main__":
    main()
