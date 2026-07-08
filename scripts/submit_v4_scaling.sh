#!/bin/bash
#SBATCH --job-name=cplex
#SBATCH --partition=amdfast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/personal/%u/autostore/order_station_assign/logs/v6_CPonly_1_olddatagen/logs/%A_%a.out

# ALL bash logic goes AFTER the #SBATCH directives
WORKDIR=/mnt/personal/$USER/autostore/order_station_assign
cd "$WORKDIR" || exit 1

# Check for --overwrite flag passed to sbatch
OVERWRITE_FLAG=""
for arg in "$@"; do
    if [ "$arg" = "--overwrite" ]; then
        OVERWRITE_FLAG="--overwrite"
    fi
done

echo "==========================================================="
echo "Starting job $SLURM_JOB_ID ($SLURM_JOB_NAME)"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $SLURM_JOB_NODELIST"
echo "Using partition: $SLURM_JOB_PARTITION"
echo "Overwrite: ${OVERWRITE_FLAG:-no}"
echo "Current directory: $(pwd)"
echo "==========================================================="

ml --ignore-cache purge
ml CPLEX/22.1.0-foss-2022a
ml plotly.py/5.12.0-GCCcore-11.3.0

RESULT_DIR=$WORKDIR/logs/$OUT_FOLDER_PREFIX
mkdir -p "$RESULT_DIR"

python benchmark_v4_single.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --timelimit "$TIMELIMIT" \
    --output-dir "$RESULT_DIR" \
    --model-version "$MODEL_VERSION" \
    --mode "${MODE:-cp}" \
    --heuristic-module "${HEURISTIC_MODULE:-autostore_heuristic}" \
    $OVERWRITE_FLAG
