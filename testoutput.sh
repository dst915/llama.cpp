#!/bin/bash

# 当任何命令失败时，立即退出脚本
set -e

# --- 配置区 ---
# 您的 llama.cpp 项目根目录
PROJECT_ROOT="/disk/doust/vllm_qw/llama/llama.cpp"
# 您要用于测试的 GGUF 模型文件的完整路径
MODEL_PATH="/disk/doust/models/Qwen1.5-4B-Chat-GGUF/qwen/Qwen1.5-4B-Chat-GGUF/qwen1_5-4b-chat-q5_k_m.gguf"
# 日志文件名
LOG_FILE="concurrency_test_log.txt"
# 硬件监控数据文件名
GPU_STATS_FILE="gpu_stats.csv"
CPU_STATS_FILE="cpu_stats.csv" # 改为 .csv
# 工具调用测试数据集文件名
DATASET_FILE="concurrency_tasks.json"

# --- 测试参数 ---
# 【优化】降低了默认并发和请求数，方便快速调试。您可以为真实压力测试调高这些值。
CONCURRENCY=5
TOTAL_REQUESTS=20
# 服务器地址和端口
HOST="127.0.0.1"
PORT="8080"

# --- 脚本函数 ---

# 【新增】彩色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 【优化】带颜色和时间戳的日志函数
log_ts() {
    # 用法: log_ts "$GREEN" "这是一条成功信息"
    local color=$1
    shift
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${color}$*${NC}"
}

# 创建工具调用数据集的函数
create_dataset() {
    log_ts "$YELLOW" "正在创建工具调用测试数据集: $DATASET_FILE..."
cat <<'EOF' > "$DATASET_FILE"
[
    {
        "id": "parallel_task",
        "prompt": "<|im_start|>system\nYou are a helpful shopping assistant.\n<|im_end|>\n<|im_start|>user\nQuickly, I need to know the stock for the RTX-4090 and at the same time, find details for the 'Super Air Fryer XL'.\n<|im_end|>\n<|im_start|>assistant"
    },
    {
        "id": "sequential_task",
        "prompt": "<|im_start|>system\nYou are a helpful shopping assistant.\n<|im_end|>\n<|im_start|>user\nWhat was my latest order ID, and can you check its shipping status?\n<|im_end|>\n<|im_start|>assistant"
    },
    {
        "id": "single_task",
        "prompt": "<|im_start|>system\nYou are a helpful shopping assistant.\n<|im_end|>\n<|im_start|>user\nI need a backpack for my laptop.\n<|im_end|>\n<|im_start|>assistant"
    }
]
EOF
    log_ts "$GREEN" "数据集创建成功。"
}

# 显示进度条的函数
progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed_width=$((width * percentage / 100))
    local remaining_width=$((width - completed_width))

    local bar="["
    for ((i=0; i<completed_width; i++)); do bar+="#"; done
    if [ $completed_width -lt $width ]; then bar+=">"; fi
    for ((i=0; i<remaining_width; i++)); do bar+="."; done
    bar+="]"

    printf "\r${GREEN}进度: %s %d%% (%d/%d)${NC}" "$bar" "$percentage" "$current" "$total"
}


# --- 脚本主体 ---

log_ts "$BLUE" "--- 1. 检查依赖项 ---"
for cmd in curl top nvidia-smi bc jq; do
    if ! command -v $cmd &> /dev/null; then
        log_ts "$RED" "错误: 核心依赖项 '$cmd' 未安装。"
        exit 1
    fi
done
log_ts "$GREEN" "所有依赖项 (curl, top, nvidia-smi, bc, jq) 已找到。"
echo ""

log_ts "$BLUE" "--- 2. 进入项目根目录并准备测试数据 ---"
cd "$PROJECT_ROOT"
create_dataset

prompts=()
while IFS= read -r -d '' prompt; do
    prompts+=("$prompt")
done < <(jq -j '.[] | .prompt, "\u0000"' "$DATASET_FILE")
num_prompts=${#prompts[@]}
log_ts "$GREEN" "已从数据集中加载 $num_prompts 个工具调用 prompts。"
echo ""

SERVER_EXECUTABLE=$(find ./build -name llama-server -type f -executable | head -n 1)
if [ -z "$SERVER_EXECUTABLE" ]; then
    log_ts "$RED" "错误: 在 './build' 目录中找不到名为 'llama-server' 的可执行文件。"
    exit 1
fi
log_ts "$GREEN" "成功找到服务器程序: $SERVER_EXECUTABLE"
echo ""

> "$LOG_FILE"
> "$GPU_STATS_FILE"
> "$CPU_STATS_FILE"

log_ts "$BLUE" "--- 3. 在后台启动 llama.cpp 服务器 (启用连续批处理和并行槽) ---"
log_ts "$YELLOW" "服务器的实时输出将显示如下，同时也会被保存到 $LOG_FILE"
echo "-------------------- SERVER LOG START --------------------"

# **【核心修复】** 添加 -np $CONCURRENCY 参数，让服务器以真正的并行模式运行
("$SERVER_EXECUTABLE" -m "$MODEL_PATH" --n-gpu-layers 99 -c 4096 --host "$HOST" --port "$PORT" --cont-batching -np "$CONCURRENCY" 2>&1 | tee "$LOG_FILE") &
SERVER_PID=$!

trap "echo -e \"\n[$(date '+%Y-%m-%d %H:%M:%S')] ${YELLOW}测试结束，正在关闭服务器和监控进程...${NC}\"; kill $SERVER_PID $NVIDIA_SMI_PID $TOP_PID 2>/dev/null" EXIT

log_ts "$YELLOW" "等待服务器启动并响应健康检查... (最长等待90秒)"
max_wait=90
waited=0
until curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://${HOST}:${PORT}/health" | grep -q 200; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "-------------------- SERVER LOG END --------------------"
        log_ts "$RED" "错误: 服务器进程 (PID: $SERVER_PID) 意外退出！"
        log_ts "$RED" "以上为服务器崩溃前的最后输出。"
        exit 1
    fi
    if [ $waited -ge $max_wait ]; then
        echo "-------------------- SERVER LOG END --------------------"
        log_ts "$RED" "错误: 服务器仍在运行，但在 ${max_wait} 秒内未能响应健康检查。"
        exit 1
    fi
    
    remaining_time=$((max_wait - waited))
    printf "\r${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] 正在等待... 剩余时间: %2d 秒${NC}" "$remaining_time"
    sleep 2
    waited=$((waited + 2))
done
echo ""
echo "-------------------- SERVER LOG END --------------------"
log_ts "$GREEN" "服务器已启动并准备就绪，PID: $SERVER_PID"
echo ""

log_ts "$BLUE" "--- 4. 开始并发性能测试与硬件监控 ---"
log_ts "$YELLOW" "并发等级: $CONCURRENCY | 请求总数: $TOTAL_REQUESTS"
log_ts "$YELLOW" "服务器日志已保存到: $LOG_FILE"

mkdir -p curl_responses
rm -f curl_responses/*

# 【优化】为硬件监控添加时间戳并精简输出
log_ts "$YELLOW" "正在后台启动硬件监控..."
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$GPU_STATS_FILE" &
NVIDIA_SMI_PID=$!

(while kill -0 $SERVER_PID 2>/dev/null; do
    top -b -n 1 | grep '%Cpu(s)' | awk -v date="$(date '+%Y-%m-%d %H:%M:%S')" '{printf "%s,%.2f\n", date, 100-$8}' >> "$CPU_STATS_FILE"
    sleep 1
done) &
TOP_PID=$!

start_time=$(date +%s.%N)

for i in $(seq 1 "$TOTAL_REQUESTS"); do
    random_prompt_index=$(( RANDOM % num_prompts ))
    prompt_to_send=${prompts[$random_prompt_index]}
    json_payload=$(jq -n --arg p "$prompt_to_send" '{"prompt": $p, "n_predict": 128, "stream": false}')

    curl -s -X POST "http://${HOST}:${PORT}/completion" \
        --max-time 120 \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        -o "curl_responses/response_${i}.json" &

    if (( i % CONCURRENCY == 0 )); then
        wait
        progress_bar $i $TOTAL_REQUESTS
    fi
done

wait
progress_bar $TOTAL_REQUESTS $TOTAL_REQUESTS
echo -e "\n$(log_ts "$GREEN" "所有 $TOTAL_REQUESTS 个请求已发送并完成。")"
echo ""

end_time=$(date +%s.%N)

kill $NVIDIA_SMI_PID $TOP_PID

# --- 5. 计算并显示结果 ---
total_time=$(echo "$end_time - $start_time" | bc)
throughput=$(echo "scale=2; $TOTAL_REQUESTS / $total_time" | bc)
avg_latency=$(echo "scale=2; $total_time * 1000 / $TOTAL_REQUESTS" | bc)

# 【优化】从新的精简日志中提取数据
avg_cpu_util=$(awk -F',' '{sum += $2} END { if (NR > 0) printf "%.2f", sum/NR }' "$CPU_STATS_FILE")
avg_gpu_util=$(awk -F', ' '{sum += $2} END { if (NR > 0) printf "%.2f", sum/NR }' "$GPU_STATS_FILE")
max_gpu_mem=$(awk -F', ' 'BEGIN{max=0} {if ($3 > max) max=$3} END{print max}' "$GPU_STATS_FILE")

echo "========================================================================"
log_ts "$BLUE" "综合性能测试分析 (工具调用场景)"
echo "========================================================================"
echo -e "${GREEN}>> 服务性能指标${NC}"
echo "------------------------------------------------------------------------"
echo "总耗时:          ${total_time} 秒"
echo "完成的总请求数:  ${TOTAL_REQUESTS}"
echo "并发等级:        ${CONCURRENCY}"
echo "吞吐量 (RPS):    ${throughput} 请求/秒"
echo "平均响应时间 (RT): ${avg_latency} 毫秒/请求"
echo ""
echo -e "${GREEN}>> 硬件资源利用率 (测试期间)${NC}"
echo "------------------------------------------------------------------------"
echo "平均 CPU 利用率: ${avg_cpu_util} %"
echo "平均 GPU 利用率: ${avg_gpu_util} %"
echo "峰值 GPU 显存占用: ${max_gpu_mem} MiB"
echo ""
echo -e "${YELLOW}>> 诊断信息${NC}"
echo "------------------------------------------------------------------------"
failed_requests=$(find curl_responses -type f -empty | wc -l)
if [ "$failed_requests" -gt 0 ]; then
    log_ts "$RED" "警告: 检测到 ${failed_requests} 个请求失败或超时 (响应为空)。"
else
    log_ts "$GREEN" "所有 curl 请求均已成功收到响应。"
fi
if grep -q -iE "error|failed|invalid" "$LOG_FILE"; then
    log_ts "$RED" "警告: 在服务器日志 '${LOG_FILE}' 中检测到潜在错误，相关行如下:"
    grep -iE "error|failed|invalid" "$LOG_FILE" | tail -n 10
else
    log_ts "$GREEN" "服务器日志中未检测到明显错误。"
fi
echo "========================================================================"

