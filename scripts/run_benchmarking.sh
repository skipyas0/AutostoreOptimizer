#!/bin/bash

# ===========================================================
# KONFIGURACE
# ===========================================================
export OUT_FOLDER_PREFIX="v6_CPonly_1_olddatagen" # COPY to submit_v4_scaling.sh .out path
#export OUT_FOLDER_PREFIX="v5_scaling_single_2_100orders" # COPY to submit.sh .out path
#export MODEL_VERSION="v4"
export MODEL_VERSION="v5"
export MODE="cp"              # cp | heuristic | warmstart
#export HEURISTIC_MODULE="heuristic_rdi_sgc_best_score"
ENABLE_OVERWRITE=true
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TIMELIMIT=1800
MAX_SEEDS=5
#LABEL="${1:-"T=1800s"}"

WORKDIR="/mnt/personal/$USER/autostore/order_station_assign"
cd "$WORKDIR" || exit 1
OVERWRITE_FLAG=""
[ "$ENABLE_OVERWRITE" = true ] && OVERWRITE_FLAG="--overwrite"

# Spočítat očekávaný počet předem, potřeba pro krok 1 i 4
MAX_IDX=$(python benchmark_v4_single.py --print-count | tail -n 1 | awk '{print $1}')
EXPECTED_COUNT=$((MAX_IDX + 1))

# ===========================================================
# MENU
# ===========================================================
echo "Vyberte akci:"
echo "0) Spustit vše (Do all)"
echo "1) Sputit benchmark tasky"
echo "2) Najít nejlepší partition (Find best partition)"
echo "3) Zkontrolovat výstup a agregovat (Check output and aggregate)"
read -rp "Volba (0-3): " CHOICE

# ===========================================================
# 1. ANALÝZA A VÝBĚR PARTITIONY
# ===========================================================
if [[ "$CHOICE" == "0" || "$CHOICE" == "2" || "$CHOICE" == "1" ]]; then
    echo -e "\n=== 1. Výpočet parametrů a hledání partitiony ==="
    BEST_PART=$(/mnt/personal/kolarj55/autostore/order_station_assign/logs/slurm_advisor.sh \
      --cpus-per-task=1 --mem=3G --time=00:40:00 --array-size=$EXPECTED_COUNT | awk '/Best Partition:/ {print $4}')
    BEST_PART=$(echo "$BEST_PART" | sed 's/\x1b\[[0-9;]*m//g' | tr -d '[:space:]')
    if [ -z "$BEST_PART" ]; then
        echo "Slurm advisor nenalezl vhodnou partition."
        exit 1
    fi
    echo "Vybrána partition: $BEST_PART (Očekávaný počet úloh: $EXPECTED_COUNT)"
fi

# ===========================================================
# 2. ODESLÁNÍ DO FRONTY & 3. LIVE MONITOR
# ===========================================================
if [[ "$CHOICE" == "0" || "$CHOICE" == "1" ]]; then
    echo -e "\n=== 2. Odesílání Job Array do Slurmu ==="
    JOB_ID=$(sbatch --parsable -p "$BEST_PART" --array=0-"$MAX_IDX" submit_v4_scaling.sh $OVERWRITE_FLAG)
    if ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "Chyba: sbatch selhal. Job ID nebylo získáno."
        exit 1
    fi
    echo "Úlohy odeslány pod ID: $JOB_ID"

    echo -e "\n=== 3. Live Monitor ==="
    MAX_ROWS=10
    SQ_FMT="%.13i %.9P %.8j %.8T %.7M %.4D %.15R"

    printf '\033[?25l' # Hides cursor
    trap "printf '\033[?25h'; exit" INT TERM # Restores cursor on exit

    LAST_PRINT_COUNT=0

    while true; do
        OUTPUT=$(squeue -j "$JOB_ID" -o "$SQ_FMT" 2>/dev/null)
        LINES=$(echo "$OUTPUT" | wc -l)
        if [ "$LINES" -lt 2 ]; then
            # Move up 1 line and clear it
            for ((i=0; i<LAST_PRINT_COUNT; i++)); do printf '\033[1A\033[K'; done
            break
        fi
        TO_PRINT=$(echo "$OUTPUT" | head -n "$MAX_ROWS")
        PRINT_COUNT=$(echo "$TO_PRINT" | wc -l)
        if [ "$LAST_PRINT_COUNT" -gt 0 ]; then
            # Move cursor up N lines
            printf "\033[%dA" "$LAST_PRINT_COUNT"
        fi
        printf '\033[J'
        echo "$TO_PRINT" | sed 's/$/\x1b[K/'
        LAST_PRINT_COUNT="$PRINT_COUNT"
        sleep 1
    done

    printf '\033[?25h' # Restores cursor
fi

# ===========================================================
# 4. KONTROLA A AGREGACE
# ===========================================================
if [[ "$CHOICE" == "0" || "$CHOICE" == "3" ]]; then
    echo -e "\n=== 4. Kontrola výsledků a agregace ==="
    ACTUAL_COUNT=$(ls logs/$OUT_FOLDER_PREFIX/*.json 2>/dev/null | wc -l)

    if [ "$ACTUAL_COUNT" -lt "$EXPECTED_COUNT" ]; then
        echo "CHYBA: Očekáváno $EXPECTED_COUNT souborů, nalezeno $ACTUAL_COUNT."
        exit 1
    fi
    echo "Očekáváných $ACTUAL_COUNT/$EXPECTED_COUNT souborů nalezeno."

    OUT_DIR="results/${OUT_FOLDER_PREFIX}"
    ml plotly.py/6.5.0-GCCcore-14.3.0
    ml matplotlib
    python aggregate_v4_scaling.py \
        --input-dir logs/$OUT_FOLDER_PREFIX \
        --output-dir "$OUT_DIR" \
        --experiment-label "$LABEL"\
        --max-seeds "$MAX_SEEDS"\
        --model-version "$MODEL_VERSION"

    echo "Hotovo! Výsledky uloženy v: $OUT_DIR"
fi