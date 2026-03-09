#!/bin/bash

# --- Configuration ---
# YAML_PATH="/home/d00827726/codes/cann-recipes-infer-7B/models/pangu/config/openpangu_v5_7b.yaml"
YAML_PATH="/home/d00827726/codes/cann-recipes-infer-7B/models/pangu/config/openpangu_v5_7b_mxfp8.yaml"
CMD_DIR="/home/d00827726/codes/cann-recipes-infer-7B/models/pangu"
BATCH_SIZES=(1 2 32 64)
LOG_FILE="experiment_results_$(date +%Y%m%d_%H%M%S).log"

echo "Starting experiments at $(date)" | tee -a "$LOG_FILE"

# --- Loop through Batch Sizes ---
for BS in "${BATCH_SIZES[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "----------------------------------------------------" | tee -a "$LOG_FILE"
    echo "[$TIMESTAMP] Testing Batch Size: $BS" | tee -a "$LOG_FILE"
    echo "----------------------------------------------------" | tee -a "$LOG_FILE"

    # 1. Update the YAML file using sed
    # Matches 'batch_size: [number]' and replaces it with the current $BS
    sed -i "s/batch_size: [0-9]*/batch_size: $BS/" "$YAML_PATH"

    # 2. Run the inference script
    cd "$CMD_DIR" || exit
    
    # Capture both stdout and stderr into the log file
    # We use a subshell to prefix the specific BS output for easier searching later
    {
        echo "--- START OUTPUT FOR BS=$BS ---"
        bash infer.sh
        echo "--- END OUTPUT FOR BS=$BS ---"
        echo -e "\n"
    } >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "[$BS] Finished successfully." | tee -a "$LOG_FILE"
    else
        echo "[$BS] Failed. Check log for details." | tee -a "$LOG_FILE"
    fi
done

echo "All experiments completed. Logs saved to: $CMD_DIR/$LOG_FILE"
