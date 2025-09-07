// src/interrupt_manager.cpp
#include "interrupt_manager.h"
#include <iostream>
#include <vector>

// 修正构造函数，初始化 m_vocab
InterruptManager::InterruptManager(ThreadSafeQueue<FunctionResult>& result_queue, const llama_vocab* vocab)
    : m_result_queue(result_queue), m_vocab(vocab), m_critical_section(false) {}

void InterruptManager::set_critical_section(bool status) {
    m_critical_section = status;
}

// 修正 tokenization 的逻辑
std::vector<llama_token> InterruptManager::get_pending_interrupt() {
    if (!m_critical_section && !m_result_queue.empty()) {
        FunctionResult result;
        if (m_result_queue.try_pop(result)) {
            std::string cml_interrupt = "\n[INTR] " + result.identifier + " [HEAD] " + result.value + " [END]\n";
            std::cerr << "\n[Interrupt Mgr] Injecting: " << cml_interrupt;

            // 正确的 tokenization 方式
            std::vector<llama_token> tokens_list(cml_interrupt.length());
            int n_tokens = llama_tokenize(m_vocab, cml_interrupt.c_str(), cml_interrupt.length(), tokens_list.data(), tokens_list.size(), false, false);

            if (n_tokens < 0) {
                std::cerr << "\n[Interrupt Mgr] Error: Tokenization failed." << std::endl;
                return {};
            }

            tokens_list.resize(n_tokens);
            return tokens_list;
        }
    }
    return {};
}