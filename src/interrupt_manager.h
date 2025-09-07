// src/interrupt_manager.h
#pragma once

#include "async_executor.h"
#include "llama.h"
#include <vector>
#include <atomic>

class InterruptManager {
public:
    InterruptManager(ThreadSafeQueue<FunctionResult>& result_queue, const llama_vocab* vocab); // 确保构造函数参数是 vocab
    void set_critical_section(bool status);
    std::vector<llama_token> get_pending_interrupt();

private:
    ThreadSafeQueue<FunctionResult>& m_result_queue;
    const llama_vocab* m_vocab; // 确保成员是 m_vocab
    std::atomic<bool> m_critical_section;
};