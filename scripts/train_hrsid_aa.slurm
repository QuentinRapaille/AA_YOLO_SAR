#!/usr/bin/env bash

#SBATCH --job-name=AAyolo_HRSID_A
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Workdir: $(pwd)"
echo "Host: $(hostname)"
echo "Date: $(date)"

module purge
module load anaconda3/2023.09-0/none-none

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "conda introuvable: verifier les modules disponibles avec 'module avail'"
  exit 1
fi

conda activate aa-yolo-sar

python -V
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

python train.py \
  --workers 8 \
  --batch-size 16 \
  --data data/hrsid.yaml \
  --img-size 640 640 \
  --iou-thres 0.05 \
  --epochs 400 \
  --cfg cfg/training/AA-yolov7-tiny.yaml \
  --hyp data/hyp.scratch.AA_yolo.yaml \
  --name train_hrsid_optionA \
  --single-cls
