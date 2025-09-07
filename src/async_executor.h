// src/async_executor.h
#pragma once

#include "thread_safe_queue.h"
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>

struct FunctionCall {
    std::string identifier;
    std::string code;
};

struct FunctionResult {
    std::string identifier;
    std::string value;
};

class AsyncExecutor {
public:
    AsyncExecutor(int num_workers, ThreadSafeQueue<FunctionResult>& result_queue);
    ~AsyncExecutor();
    void submit(FunctionCall call);
    bool is_idle(); // <-- 新增

private:
    void worker_loop();
    std::vector<std::thread> m_workers;
    ThreadSafeQueue<FunctionCall> m_task_queue;
    ThreadSafeQueue<FunctionResult>& m_result_queue;
    std::atomic<bool> m_stop;
    std::atomic<int> m_active_tasks; // <-- 新增

};