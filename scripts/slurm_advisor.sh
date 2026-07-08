#!/bin/bash

# Defaults
REQ_CPU=1
REQ_MEM=4
REQ_GPU=0
REQ_TIME="0"
REQ_ARRAY=1

# Smart parsing for spaces and '='
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpus-per-task) REQ_CPU="$2"; shift 2 ;;
        --cpus-per-task=*) REQ_CPU="${1#*=}"; shift 1 ;;
        --mem) REQ_MEM="$2"; shift 2 ;;
        --mem=*) REQ_MEM="${1#*=}"; shift 1 ;;
        --gres) REQ_GPU="$2"; shift 2 ;;
        --gres=*) REQ_GPU="${1#*=}"; shift 1 ;;
        --time) REQ_TIME="$2"; shift 2 ;;
        --time=*) REQ_TIME="${1#*=}"; shift 1 ;;
        --array-size) REQ_ARRAY="$2"; shift 2 ;;
        --array-size=*) REQ_ARRAY="${1#*=}"; shift 1 ;;
        *) shift 1 ;;
    esac
done

# Smart memory conversion to GB
if [[ $REQ_MEM =~ [Mm] ]]; then
    REQ_MEM=$(echo "$REQ_MEM" | tr -d 'A-Za-z')
    REQ_MEM=$((REQ_MEM / 1024))
    [ $REQ_MEM -eq 0 ] && REQ_MEM=1 # minimum 1GB
else
    REQ_MEM=$(echo "$REQ_MEM" | tr -d 'A-Za-z')
fi

echo -e "\n=== Cluster Overview ==="
showpartitions

echo -e "\n=== 🔍 Slurm Smart Advisor ==="
echo "Requested: CPU=$REQ_CPU, RAM=${REQ_MEM}G, GPU=$REQ_GPU, TIME=$REQ_TIME, ARRAY=$REQ_ARRAY"

rci_load | awk -v cpu="$REQ_CPU" -v mem="$REQ_MEM" -v gpu="$REQ_GPU" -v time_str="$REQ_TIME" -v array_size="$REQ_ARRAY" '
BEGIN {
    # Convert time to minutes
    req_mins = 0
    if (time_str != "0") {
        d = 0; t_str = time_str
        if (index(t_str, "-") > 0) { split(t_str, arr, "-"); d = arr[1]; t_str = arr[2] }
        n = split(t_str, t, ":")
        if (n == 1) req_mins = d*1440 + t[1]
        else if (n == 2) req_mins = d*1440 + t[1]
        else if (n == 3) req_mins = d*1440 + t[1]*60 + t[2]
    }

    p["cpufast"]="382 1484 0 240 8"; p["gpufast"]="70 384 4 240 8"
    p["cpu"]="382 1484 0 1440 4"; p["gpu"]="70 384 4 1440 4"
    p["smp"]="382 1484 0 1440 8"
    p["cpulong"]="46 384 0 4320 2"; p["gpulong"]="70 384 4 4320 2"
    p["cpuextralong"]="46 384 0 60480 1"; p["gpuextralong"]="70 384 4 30240 1"
    p["deadline"]="382 1484 0 1440 10"; p["gpudeadline"]="70 384 4 1440 10"
    p["amdfast"]="126 1024 0 240 8"; p["amdgpufast"]="252 1024 8 240 8"
    p["amd"]="126 1024 0 1440 4"; p["amdgpu"]="252 1024 8 1440 4"
    p["amdlong"]="126 1024 0 4320 2"; p["amdgpulong"]="252 1024 8 4320 2"
    p["amdextralong"]="126 1024 0 60480 1"; p["amdgpuextralong"]="252 1024 8 30240 1"
    p["amddeadline"]="126 1024 0 1440 10"; p["amdgpudeadline"]="252 1024 8 1440 10"
    p["h200fast"]="124 2304 8 240 8"; p["h200"]="124 2034 8 1440 4"
    p["h200long"]="124 2034 8 4320 2"; p["h200extralong"]="124 2034 8 30240 1"
    p["interactive"]="128 512 4 4320 10"; p["ipu"]="124 1024 0 1440 10"

    printf "%-18s %-8s %-8s %-8s %-11s %-9s %-8s %-8s %-10s\n", "PARTITION", "FREE_CPU", "FREE_MEM", "FREE_GPU", "CONCURRENT", "PEND_JOBS", "MAX_MINS", "PRIORITY", "SCORE"
    print "-------------------------------------------------------------------------------------------"
    best_part = ""; best_score = -99999
}

/^Partition/ { flag=1; next }
/^====/ && flag==1 { next }
/^$/ { flag=0 }

flag==1 && $1 != "CPU_A,MEM_A,GPU_A" {
    part=$1; cpu_f=$4; mem_f=$9; gpu_f=$14; jobs_p=$18

    full = part
    if (part == "amdextra") full = "amdextralong"
    if (part == "amdgpuex") full = "amdgpuextralong"
    if (part == "amdgpufa") full = "amdgpufast"
    if (part == "amdgpulo") full = "amdgpulong"
    if (part == "cpuextra") full = "cpuextralong"
    if (part == "gpuextra") full = "gpuextralong"

    if (!(full in p) || full ~ /\*/) next

    split(p[full], lim, " ")
    cpu_m = lim[1]; mem_m = lim[2]; gpu_m = lim[3]; part_limit = lim[4]; prio = lim[5]

    if (req_mins > 0 && req_mins > part_limit) next
    if (gpu == 0 && gpu_m > 0) next

    if (cpu <= cpu_m && mem <= mem_m && gpu <= gpu_m) {

        # Calculate how many tasks from the array can physically fit in free resources right now
        max_by_cpu = (cpu > 0) ? int(cpu_f / cpu) : 999999
        max_by_mem = (mem > 0) ? int(mem_f / mem) : 999999
        max_by_gpu = (gpu > 0) ? int(gpu_f / gpu) : 999999

        concurrent = max_by_cpu
        if (max_by_mem < concurrent) concurrent = max_by_mem
        if (max_by_gpu < concurrent) concurrent = max_by_gpu
        if (concurrent > array_size) concurrent = array_size

        can_start_now = (cpu_f >= cpu && mem_f >= mem && gpu_f >= gpu) ? 1 : 0

        if (array_size == 1) score = (can_start_now * 1000) + (prio * 100) + cpu_f + gpu_f - (jobs_p * 50)
        if (array_size > 1) score = (concurrent * 1000) + (prio * 100) + cpu_f + gpu_f - (jobs_p * 50)

        if (cpu_f > 0) {
            printf "%-18s %-8s %-8s %-8s %-11s %-9s %-8s %-8s %-10s\n", full, cpu_f, mem_f, gpu_f, concurrent, jobs_p, part_limit, prio, score
        }

        if (score > best_score) {
            best_score = score
            best_part = full
        }
    }
}
END {
    print "-------------------------------------------------------------------------------------------"
    if (best_part != "") {
        print "🏆 Best Partition: \033[1;32m" best_part "\033[0m"
        gres_arg = (gpu > 0) ? "--gres=gpu:" gpu " " : ""
        time_arg = (time_str != "0") ? "--time=" time_str " " : ""
        print "💡 Quick run:\tsalloc -p " best_part " -c " cpu " --mem=" mem "G " gres_arg time_arg
    } else {
        print "❌ No partition meets requirements or time limit."
    }
}'