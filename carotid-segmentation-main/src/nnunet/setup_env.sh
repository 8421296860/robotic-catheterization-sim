#!/usr/bin/env bash
# Source this before running nnUNet commands:
#   source setup_env.sh

export nnUNet_raw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../nnunet_data/nnUNet_raw" && pwd)"
export nnUNet_preprocessed="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../nnunet_data/nnUNet_preprocessed" && pwd)"
export nnUNet_results="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../nnunet_data/nnUNet_results" && pwd)"

echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
