#!/bin/bash

# 当任何命令失败时，立即退出脚本
set -e

# --- 配置区 ---
# 您的 llama.cpp 项目根目录
PROJECT_ROOT="/disk/doust/vllm_qw/llama/llama.cpp"
# 您要用于测试的 GGUF 模型文件的完整路径
MODEL_PATH="/disk/doust/models/Qwen1.5-4B-Chat-GGUF/qwen/Qwen1.5-4B-Chat-GGUF/qwen1_5-4b-chat-q5_k_m.gguf"
# 数据集文件名 (脚本会自动创建此文件)
DATASET_FILE="ecommerce_tasks.json"
# llama-cli 可执行文件的路径
EXECUTABLE="./build/bin/llama-cli"
# 日志文件名
LOG_FILE="performance_test_log.txt"

# --- 脚本主体 ---

echo "--- 1. 检查依赖项 ---"
# 检查 jq 是否安装
if ! command -v jq &> /dev/null; then
    echo "错误: 依赖项 'jq' 未安装。"
    echo "jq 是一个处理 JSON 文件的命令行工具，本脚本需要用它来解析数据集。"
    echo ""
    echo "--- 安装选项 ---"
    echo "1. [推荐] 如果您有管理员 (sudo) 权限:"
    echo "   - Ubuntu/Debian: sudo apt-get update && sudo apt-get install jq"
    echo "   - CentOS/RHEL:   sudo yum install jq"
    echo ""
    echo "2. 如果您没有管理员权限，可以进行本地安装 (推荐):"
    echo "   a) 创建一个本地 bin 目录 (如果不存在):"
    echo "      mkdir -p ~/bin"
    echo "   b) 下载 jq 的 Linux 64位 二进制文件到该目录:"
    echo "      wget -O ~/bin/jq https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-linux-amd64"
    echo "   c) 赋予它执行权限:"
    echo "      chmod +x ~/bin/jq"
    echo "   d) 将该目录添加到您的 PATH 环境变量中 (将此行加入 ~/.bashrc 或 ~/.zshrc 的末尾):"
    echo "      export PATH=\"\$HOME/bin:\$PATH\""
    echo "   e) 重新加载您的 shell 配置或开启一个新的终端来使配置生效:"
    echo "      source ~/.bashrc"
    echo ""
    exit 1
fi
echo "依赖项 'jq' 已找到。"
echo ""

echo "--- 2. 进入项目根目录 ---"
cd "$PROJECT_ROOT"

# --- 3. 使用 Here Document 动态创建数据集文件 ---
# 脚本会将下面的 JSON 内容写入到 $DATASET_FILE
cat <<'EOF' > "$DATASET_FILE"
[
    {
        "id": "parallel_task",
        "description": "测试可以并行执行的两个独立工具调用",
        "prompt": "<|im_start|>system\nYou are a helpful shopping assistant.\n<|im_end|>\n<|im_start|>user\nQuickly, I need to know the stock for the RTX-4090 and at the same time, find details for the 'Super Air Fryer XL'.\n<|im_end|>\n<|im_start|>assistant\n[CALL] stock_check [HEAD] get_stock_by_sku('RTX-4090') [END][CALL] product_details [HEAD] get_product_details('Super Air Fryer XL') [END]"
    },
    {
        "id": "sequential_task",
        "description": "测试必须按顺序执行的两个依赖工具调用",
        "prompt": "<|im_start|>system\nYou are a helpful shopping assistant.\n<|im_end|>\n<|im_start|>user\nWhat was my latest order ID, and can you check its shipping status?\n<|im_end|>\n<|im_start|>assistant\n[CALL] find_order [HEAD] get_latest_order_id() [END]"
    },
    {
        "id": "single_task",
        "description": "测试单个工具调用作为基线",
        "prompt": "<|im_start|>system\nYou are a helpful shopping assistant.\n<|im_end|>\n<|im_start|>user\nI need a backpack for my laptop.\n<|im_end|>\n<|im_start|>assistant\n[CALL] search_job [HEAD] query_products(category='backpack', tags=['laptop']) [END]"
    }
]
EOF

echo "数据集 '$DATASET_FILE' 已在 $(pwd) 中成功创建。"
echo ""

# --- 4. 准备并执行测试 ---
echo "性能测试日志将保存在: $(pwd)/${LOG_FILE}"
# 清空旧日志
> "$LOG_FILE"

# 检查可执行文件是否存在
if [ ! -f "$EXECUTABLE" ]; then
    echo "错误: 可执行文件 '$EXECUTABLE' 未找到。请先运行编译脚本。"
    exit 1
fi

# 使用 jq 解析 JSON 数据集并循环执行测试
jq -c '.[]' "$DATASET_FILE" | while read -r task; do
    id=$(jq -r '.id' <<< "$task")
    description=$(jq -r '.description' <<< "$task")
    prompt=$(jq -r '.prompt' <<< "$task")

    echo "========================================================================" | tee -a "$LOG_FILE"
    echo "正在运行任务: [$id] - $description" | tee -a "$LOG_FILE"
    echo "========================================================================" | tee -a "$LOG_FILE"

    # 使用 /usr/bin/time -v 来获取详细的计时信息，特别是 'Elapsed (wall clock) time'
    # 我们将 stderr(2) 和 stdout(1) 都重定向到日志文件以捕获所有输出
    {
        /usr/bin/time -v "$EXECUTABLE" -m "$MODEL_PATH" \
            --n-gpu-layers 99 \
            -n 512 \
            --ctx-size 4096 \
            -p "$prompt" >> "$LOG_FILE" 2>&1
    } 2>> "$LOG_FILE"

    echo "任务 [$id] 完成。"
    echo "" | tee -a "$LOG_FILE"
done

# --- 5. 结果分析 ---
echo "========================================================================"
echo "性能测试分析"
echo "========================================================================"
echo "所有任务已完成。正在从日志 '$LOG_FILE' 中提取执行时间..."

# 从日志中提取并行任务和串行任务的执行时间
# 【修正】将 grep 的上下文行数从 -A 25 增加到 -A 200，确保能捕获到日志末尾的计时信息
parallel_time=$(grep -A 600 "正在运行任务: \[parallel_task\]" "$LOG_FILE" | grep "Elapsed (wall clock) time" | tail -1 | awk '{print $NF}')
sequential_time=$(grep -A 600 "正在运行任务: \[sequential_task\]" "$LOG_FILE" | grep "Elapsed (wall clock) time" | tail -1 | awk '{print $NF}')

echo ""
echo "并行任务 (受益于异步) 总耗时: ${parallel_time}"
echo "串行任务 (不受益于异步) 总耗时: ${sequential_time}"
echo ""
echo "分析:"
echo "1. 请观察 [parallel_task] 的日志。你应该能看到两个 '[Executor] Starting...' 几乎同时出现，这证明了函数调用是并行执行的。"
echo "2. 并行任务的总耗时应该约等于 '最长的那个API延迟' + 'LLM生成和处理时间'。"
echo "3. 一个理论上的同步执行器完成并行任务的时间约等于 'API延迟1 + API延迟2 + LLM时间'，因此异步版本显著节省了等待时间。"
echo "4. 完整的日志和计时细节已保存在 '$LOG_FILE' 文件中供详细分析。"

