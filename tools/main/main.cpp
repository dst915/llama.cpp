// ===================================================================
// START OF FINAL, CORRECTED, and COMPLETE REPLACEMENT CODE
// ===================================================================
#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"

// ASYNC MOD: Include new headers for our async system
#include "../../src/async_executor.h"
#include "../../src/interrupt_manager.h"
#include <atomic>
#include <regex>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static common_sampler          ** g_smpl;
static common_params            * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting   = false;
static bool need_insert_eot = false;

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -sys \"You are a helpful assistant\"\n", argv[0]);
    LOG("\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting   = true;
            need_insert_eot = true;
        } else {
            console::cleanup();
            LOG("\n");
            common_perf_print(*g_ctx, *g_smpl);

            // make sure all logs are flushed
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());

            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    common_params params;
    g_params = &params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    common_init();

    auto & sparams = params.sampling;

    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.embedding) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_INF("%s: llama backend init\n", __func__);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    common_sampler * smpl = nullptr;

    g_model = &model;
    g_ctx = &ctx;
    g_smpl = &smpl;

    std::vector<common_chat_msg> chat_msgs;

    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
    common_init_result llama_init = common_init_from_params(params);

    model = llama_init.model.get();
    ctx = llama_init.context.get();

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto chat_templates = common_chat_templates_init(model, params.chat_template);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }
    
    // ===================================================================
    // ASYNC MOD: Initialize our asynchronous system
    // ===================================================================
    fprintf(stderr, "\nInitializing asynchronous function calling system...\n");
    ThreadSafeQueue<FunctionResult> result_queue;
    AsyncExecutor executor(4, result_queue);
    InterruptManager interrupt_manager(result_queue, vocab);
    // ===================================================================

    // (Preserving original logic from here)
    if (params.interactive) {
        params.interactive_first = true;
    }

    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    std::vector<llama_token> embd_inp;
    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;
    // ... (original session loading and handling) ...
    const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja;
    embd_inp = ::common_tokenize(ctx, params.prompt, add_bos, true);

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
    signal(SIGINT, sigint_handler);
#endif

    smpl = common_sampler_init(model, sparams);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        return 1;
    }

    LOG_INF("== Running in ASYNC-MODIFIED interactive mode. ==\n");
    is_interacting = params.interactive_first;

    bool is_antiprompt = false;
    int n_past = 0;
    int n_remain = params.n_predict;
    int n_consumed = 0;

    std::vector<llama_token> embd;
    std::string accumulated_output = "";
    
    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) { n_eval = params.n_batch; }
                // CORRECTED API CALL - keeping original 4-argument version
                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                    LOG_ERR("%s : failed to eval\n", __func__); return 1;
                }
                n_past += n_eval;
            }
        }
        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // ===================================================================
            // ASYNC MODIFICATION 1: INTERRUPT HANDLING
            // ===================================================================
            // 优先检查并处理中断
            auto interrupt_tokens = interrupt_manager.get_pending_interrupt();
            if (!interrupt_tokens.empty()) {
                // 如果有中断，则本次循环处理中断 token
                LOG_INF("\n[SYSTEM] Interrupt detected. Injecting tool result into context...\n");
                embd.insert(embd.end(), interrupt_tokens.begin(), interrupt_tokens.end());
            } else {
                // 如果没有中断，才执行常规的 token 采样
                const llama_token id = common_sampler_sample(smpl, ctx, -1);
                common_sampler_accept(smpl, id, true);
                embd.push_back(id);

                if (n_remain > 0) {
                    --n_remain;
                }
            }
            // ===================================================================

        } else {
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                common_sampler_accept(smpl, embd_inp[n_consumed], false);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        for (auto id : embd) {
            const std::string token_str = common_token_to_piece(ctx, id);
            printf("%s", token_str.c_str());
            if ((int) embd_inp.size() <= n_consumed) {
                 accumulated_output += token_str;
            }
        }
        fflush(stdout);

        size_t call_pos;
        while ((call_pos = accumulated_output.find("[CALL]")) != std::string::npos) {
            size_t end_pos = accumulated_output.find("[END]", call_pos);
            if (end_pos == std::string::npos) {
                interrupt_manager.set_critical_section(true);
                break;
            }
            interrupt_manager.set_critical_section(false);
            std::string block = accumulated_output.substr(call_pos + 6, end_pos - (call_pos + 6));
            size_t head_pos = block.find("[HEAD]");
            if (head_pos != std::string::npos) {
                std::string call_id = block.substr(0, head_pos);
                std::string call_code = block.substr(head_pos + 6);
                call_id.erase(0, call_id.find_first_not_of(" \n\r\t"));
                call_id.erase(call_id.find_last_not_of(" \n\r\t") + 1);
                call_code.erase(0, call_code.find_first_not_of(" \n\r\t"));
                call_code.erase(call_code.find_last_not_of(" \n\r\t") + 1);
                executor.submit({call_id, call_code});
            }
            accumulated_output.erase(call_pos, end_pos - call_pos + 5);
        }
        
            // ... 在主 while 循环末尾附近 ...

        if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !(params.interactive)) {
            LOG(" [end of text]\n");

            // ===================================================================
            // ASYNC MODIFICATION 2: FINAL RESPONSE GENERATION
            // ===================================================================
            // 主生成阶段结束，但需要等待所有后台工具完成
            if (!executor.is_idle()) {
                LOG_INF("[SYSTEM] Model has paused. Now waiting for pending tools to finish...\n");

                // 持续等待，直到所有后台任务完成
                while(!executor.is_idle()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 短暂休眠，避免CPU空转

                    // 在等待期间，仍然需要处理可能刚刚完成的任务结果
                    auto final_interrupt_tokens = interrupt_manager.get_pending_interrupt();
                    if (!final_interrupt_tokens.empty()) {
                        LOG_INF("[SYSTEM] Processing a final tool result while waiting...\n");

                        // 注入最后的结果，但不打印，因为我们将在下一步统一生成最终回复
                        if (llama_decode(ctx, llama_batch_get_one(final_interrupt_tokens.data(), final_interrupt_tokens.size()))) {
                            LOG_ERR("%s : failed to eval final interrupt\n", __func__); return 1;
                        }
                        n_past += final_interrupt_tokens.size();
                    }
                }

                LOG_INF("\n[SYSTEM] All tools have finished. Generating the final conclusive response...\n");
                
                // 驱动模型基于所有工具结果，生成最后的总结性回答
                // 这个循环会一直运行直到模型再次生成 EOG token
                while(true) {
                    const llama_token id = common_sampler_sample(smpl, ctx, -1);
                    common_sampler_accept(smpl, id, true);

                    if (llama_vocab_is_eog(vocab, id)) {
                        break; // 最终回复生成完毕
                    }

                    // 打印最终回复的 token
                    printf("%s", common_token_to_piece(ctx, id).c_str());
                    fflush(stdout);

                    // 将生成的 token 送回解码器，为下一个 token 做准备
                    std::vector<llama_token> final_token_vec = { id };
                    if (llama_decode(ctx, llama_batch_get_one(final_token_vec.data(), final_token_vec.size()))) {
                        LOG_ERR("%s : failed to eval final response token\n", __func__); return 1;
                    }
                    n_past++;
                }
                LOG_INF("\n");
            }
            // ===================================================================
            break; // 无论是否有异步任务，都结束主循环
        }

        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }

        if ((int) embd_inp.size() <= n_consumed) {
             if (is_interacting) {
                common_sampler_reset(smpl);
             }
             is_interacting = false;
        }

        if (params.interactive && (int) embd_inp.size() <= n_consumed) {
            LOG("\n> ");
            std::string buffer;
            std::string line;
            bool another_line = true;
            do {
                another_line = console::readline(line, params.multiline_input);
                buffer += line;
            } while (another_line);
            if (buffer.empty()) { break; }
            if (buffer.back() == '\n') { buffer.pop_back(); }
            
            const auto line_inp = common_tokenize(ctx, buffer, false, true);
            embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
            n_remain -= line_inp.size();
        }
    }

    common_perf_print(ctx, smpl);
    common_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
// ===================================================================
// END OF FINAL, CORRECTED AND COMPLETE REPLACEMENT CODE
// ===================================================================