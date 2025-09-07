// src/async_executor.cpp
#include "async_executor.h"
#include <iostream>
#include <chrono>
#include <random>
#include <map>

// 模拟的 API 函数实现
std::string run_mock_api_call(const std::string& function_code) {
    // 模拟 2 到 5 秒的随机延迟
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(2000, 5000);
    std::this_thread::sleep_for(std::chrono::milliseconds(distrib(gen)));

    // 简陋的解析和预设的返回结果
    if (function_code.find("get_stock_by_sku") != std::string::npos) {
        return "{\"sku\": \"RTX-4090\", \"stock\": 15}";
    }
    if (function_code.find("get_product_details") != std::string::npos) {
        return "{\"name\": \"Super Air Fryer XL\", \"price\": 99.99}";
    }
    if (function_code.find("get_latest_order_id") != std::string::npos) {
        return "{\"order_id\": \"ORD-2025-98777\"}";
    }
    if (function_code.find("get_shipping_status") != std::string::npos) {
        return "{\"status\": \"Shipped\"}";
    }
    if (function_code.find("query_products") != std::string::npos) {
        return "[{\"product_id\": \"BP-LITE-GRY\"}]";
    }
    return "{\"error\": \"Unknown function\"}";
}

AsyncExecutor::AsyncExecutor(int num_workers, ThreadSafeQueue<FunctionResult>& result_queue)
    : m_result_queue(result_queue), m_stop(false), m_active_tasks(0) {
    for (int i = 0; i < num_workers; ++i) {
        m_workers.emplace_back([this] { this->worker_loop(); });
    }
}

AsyncExecutor::~AsyncExecutor() {
    m_stop = true;
    // 添加空任务来唤醒可能正在等待的线程
    for (size_t i = 0; i < m_workers.size(); ++i) {
        submit({"", ""});
    }
    for (std::thread& worker : m_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void AsyncExecutor::submit(FunctionCall call) {
    m_active_tasks++; // <-- 新增
    m_task_queue.push(call);
}

void AsyncExecutor::worker_loop() {
    while (!m_stop) {
        FunctionCall task;
        if (m_task_queue.try_pop(task)) {
            if (m_stop) return;
            std::cerr << "\n[Executor] Starting: " << task.code << std::endl;
            std::string result_value = run_mock_api_call(task.code);
            m_result_queue.push({task.identifier, result_value});
            m_active_tasks--; // <-- 新增
            std::cerr << "\n[Executor] Finished: " << task.code << " -> " << result_value << std::endl;
        }
    }
}
bool AsyncExecutor::is_idle() {
    return m_task_queue.empty() && (m_active_tasks == 0);
}