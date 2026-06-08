"""Plan, preprocess, and train nnUNetv2 on the carotid dataset."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_CarotidArtery"
NNUNET_ROOT = Path(__file__).resolve().parents[2] / "nnunet_data"


def parse_args():
    parser = argparse.ArgumentParser(description="Train nnUNetv2 on carotid dataset")
    parser.add_argument(
        "--configuration",
        "-c",
        default="2d",
        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
        help="nnUNet configuration (2d recommended for ultrasound slices)",
    )
    parser.add_argument(
        "--fold",
        default="all",
        help="Cross-validation fold: 0-4 or 'all' to train on full dataset",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only convert dataset to nnUNet format",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only run plan_and_preprocess (skip training)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip dataset prep and preprocessing, run training only",
    )
    parser.add_argument(
        "--npz",
        action="store_true",
        help="Save softmax outputs during validation (needed for ensembling)",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of CV folds for plan_and_preprocess",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Training device",
    )
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="Continue from last checkpoint",
    )
    return parser.parse_args()


def setup_env() -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str((NNUNET_ROOT / "nnUNet_raw").resolve())
    env["nnUNet_preprocessed"] = str((NNUNET_ROOT / "nnUNet_preprocessed").resolve())
    env["nnUNet_results"] = str((NNUNET_ROOT / "nnUNet_results").resolve())
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run(cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, env=env, check=True)


def main():
    args = parse_args()
    env = setup_env()

    print("nnUNet environment:")
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        print(f"  {key}={env[key]}")

    if not args.train_only and not args.preprocess_only:
        from prepare_dataset import prepare_dataset

        source = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "Common Carotid Artery Ultrasound Images"
        )
        prepare_dataset(source, Path(env["nnUNet_raw"]))

    if args.prepare_only:
        return

    if not args.train_only:
        preprocess_cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(DATASET_ID),
            "--verify_dataset_integrity",
            "-npfp",
            "8",
            "-np",
            "8",
            "-c",
            args.configuration,
        ]
        if args.num_folds != 5:
            preprocess_cmd.extend(["--num_folds", str(args.num_folds)])
        run(preprocess_cmd, env)

    if args.preprocess_only:
        return

    train_cmd = [
        "nnUNetv2_train",
        str(DATASET_ID),
        args.configuration,
        str(args.fold),
        "-device",
        args.device,
    ]
    if args.npz:
        train_cmd.append("--npz")
    if args.continue_training:
        train_cmd.append("--c")

    run(train_cmd, env)

    print(f"\nTraining finished. Results: {env['nnUNet_results']}/{DATASET_NAME}/")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
