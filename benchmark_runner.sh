# #!/bin/bash

# # === CONFIG ===
# VARIANT=$1
# if [ -z "$VARIANT" ]; then
#   echo "Usage: $0 <variant-name> (e.g., baseline, depthanything, mvs)"
#   exit 1
# fi

# DATASET_PATH="dataset/dense/ignatius"
# DEPTH_MAP_DIR="$DATASET_PATH/stereo/depth_maps"
# BACKUP_NAME="depth_maps_$(date +"%Y-%m-%dT%H-%M-%S")_${VARIANT}.tar.gz"
# MODEL_OUT="output/$VARIANT"

# # === STEP 1: BACKUP DEPTH MAPS ===
# echo "📦 Backing up depth maps to $BACKUP_NAME"
# tar -czf "$BACKUP_NAME" "$DEPTH_MAP_DIR"

# # === STEP 2: TRAINING ===
# echo "🚀 Running training for variant: $VARIANT"
# python train.py \
#   --config configs/ignatius.yaml \
#   --model_path "$MODEL_OUT" \
#   --variant_name "$VARIANT"

# echo "✅ Benchmark for '$VARIANT' complete. Output stored in $MODEL_OUT"
