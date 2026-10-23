// Cycle-level speculative decoding bridge, compiled against llama.cpp 74ade5274.
//
// Q-learning stays in Python. Python sends one decision table per verification
// cycle; the bridge looks up each per-token continue/stop decision locally so a
// decision costs no extra IPC round trip. Every drafted token returns the draft
// model's top-1 probability (its confidence) and every cycle returns draft lag.
//
// Protocol (one JSON line per reply):
//   CONFIG <max_draft> <n_thresholds> <t1> ... <tn>
//   RESET <token_limit> <prompt_hex>
//   CYCLE <draft_first 0|1> <bits|->   bits[(k-1)*n_buckets+b]=='1' -> draft token k+1
//     where b = bucket of the product of the k draft confidences so far
//   TEXT
//   PIECES <id> <id> ...                 text of each token id
//   QUIT
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Tokens = std::vector<llama_token>;
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
std::string hex(const std::string & text) {
    static const char * chars="0123456789abcdef";
    std::string result;
    for (unsigned char c:text) { result+=chars[c>>4]; result+=chars[c&15]; }
    return result;
}
std::string unhex(const std::string & text) {
    if (text.size()%2) throw std::runtime_error("Invalid hex text");
    std::string result;
    for (size_t i=0;i<text.size();i+=2) result+=char(std::stoul(text.substr(i,2),nullptr,16));
    return result;
}

struct Model {
    llama_model * model=nullptr;
    llama_context * ctx=nullptr;
    const llama_vocab * vocab=nullptr;
    int past=0;
    int layers=0;
    float single_gpu_split[1]={1.0f};
    Model(const char * path,int gpu_layers,int context,int threads) {
        auto mp=llama_model_default_params();
        mp.n_gpu_layers=gpu_layers;
        mp.tensor_split=single_gpu_split;
        mp.use_mmap=true;
        model=llama_model_load_from_file(path,mp);
        if (!model) throw std::runtime_error("Model load failed; inspect native-runtime.log");
        vocab=llama_model_get_vocab(model);
        layers=llama_model_n_layer(model);
        auto cp=llama_context_default_params();
        cp.n_ctx=context; cp.n_batch=context; cp.n_ubatch=64;
        cp.n_seq_max=1; cp.n_outputs_max=10;
        cp.n_threads=threads; cp.n_threads_batch=threads;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.no_perf=true;
        cp.offload_kqv=gpu_layers!=0;
        cp.op_offload=gpu_layers!=0;
        ctx=llama_init_from_model(model,cp);
        if (!ctx) throw std::runtime_error("Context initialization failed");
        if (llama_model_is_hybrid(model) || llama_model_is_recurrent(model))
            throw std::runtime_error("Only standard transformer models (KV-cache rollback) are supported");
    }
    ~Model() { if(ctx) llama_free(ctx); if(model) llama_model_free(model); }
    void clear() { llama_memory_clear(llama_get_memory(ctx),true); past=0; }
    void rewind(int position) {
        if(position>=past) return;
        if(!llama_memory_seq_rm(llama_get_memory(ctx),0,position,-1))
            throw std::runtime_error("KV rollback rejected; stopping rather than corrupting history");
        past=position;
    }
    void eval(const Tokens & tokens,bool all_logits) {
        if(tokens.empty()) return;
        auto batch=llama_batch_init(int(tokens.size()),0,1);
        batch.n_tokens=int(tokens.size());
        for(int i=0;i<batch.n_tokens;++i) {
            batch.token[i]=tokens[i]; batch.pos[i]=past+i;
            batch.n_seq_id[i]=1; batch.seq_id[i][0]=0;
            batch.logits[i]=all_logits || i==batch.n_tokens-1;
        }
        int result=llama_decode(ctx,batch);
        llama_batch_free(batch);
        if(result!=0) throw std::runtime_error("llama_decode failed: "+std::to_string(result));
        llama_synchronize(ctx);
        past+=int(tokens.size());
    }
    const float * logits(int batch_index) {
        const float * values=llama_get_logits_ith(ctx,batch_index);
        if(!values) throw std::runtime_error("Missing logits");
        return values;
    }
    llama_token greedy(int batch_index) {
        const float * l=logits(batch_index);
        return llama_token(std::max_element(l,l+llama_vocab_n_tokens(vocab))-l);
    }
    // Greedy token plus its softmax probability. Logits more than 20 nats below
    // the maximum contribute < 2e-9 each and are skipped.
    llama_token greedy_confidence(int batch_index,double & confidence) {
        const float * l=logits(batch_index);
        int n=llama_vocab_n_tokens(vocab);
        const float * best=std::max_element(l,l+n);
        float top=*best, floor=top-20.0f;
        double sum=0;
        for(int i=0;i<n;++i) if(l[i]>floor) sum+=std::exp(double(l[i]-top));
        confidence=1.0/sum;
        return llama_token(best-l);
    }
    Tokens tokenize(const std::string & text) {
        int size=llama_tokenize(vocab,text.data(),int(text.size()),nullptr,0,false,true);
        if(size>=0) throw std::runtime_error("Unexpected tokenization size");
        Tokens tokens(-size);
        size=llama_tokenize(vocab,text.data(),int(text.size()),tokens.data(),int(tokens.size()),false,true);
        if(size<0) throw std::runtime_error("Tokenization failed");
        tokens.resize(size); return tokens;
    }
    std::string text(const Tokens & tokens) {
        if(tokens.empty()) return "";
        int size=llama_detokenize(vocab,tokens.data(),int(tokens.size()),nullptr,0,true,false);
        if(size>=0) return "";
        std::string result(-size,'\0');
        size=llama_detokenize(vocab,tokens.data(),int(tokens.size()),result.data(),int(result.size()),true,false);
        if(size<0) throw std::runtime_error("Detokenization failed");
        result.resize(size); return result;
    }
};

struct Decision { int k; int bucket; int action; bool forced; };

int main(int argc,char ** argv) {
    std::cout<<std::setprecision(12);
    try {
        if(argc!=9) throw std::runtime_error("bridge TARGET DRAFT TARGET_LAYERS DRAFT_LAYERS CONTEXT THREADS BACKEND_ROOT CUDA_DIR");
        ggml_backend_load_all_from_path(argv[8]);
        ggml_backend_load_all_from_path(argv[7]);
        llama_backend_init();
        int context=std::stoi(argv[5]);
        bool standalone=std::getenv("LTD_STANDALONE_TARGET_ONLY") && std::string(std::getenv("LTD_STANDALONE_TARGET_ONLY"))=="1";
        Model target(argv[1],std::stoi(argv[3]),context,std::stoi(argv[6]));
        std::unique_ptr<Model> draft_owner;
        if(!standalone) draft_owner=std::make_unique<Model>(argv[2],std::stoi(argv[4]),context,std::stoi(argv[6]));
        Model * draft=draft_owner.get();
        int vocab=llama_vocab_n_tokens(target.vocab);
        if(draft && vocab!=llama_vocab_n_tokens(draft->vocab)) throw std::runtime_error("Vocabulary sizes differ");
        for(int i=0;draft && i<vocab;++i) {
            if(std::strcmp(llama_vocab_get_text(target.vocab,i),llama_vocab_get_text(draft->vocab,i))!=0)
                throw std::runtime_error("Vocabulary token IDs differ");
        }
        std::cout<<"{\"ready\":true,\"vocab_size\":"<<vocab
                 <<",\"target_total_layers\":"<<target.layers<<",\"draft_total_layers\":"<<(draft?draft->layers:0)
                 <<",\"draft_loaded\":"<<(draft?"true":"false")<<"}"<<std::endl;
        Tokens committed,output;
        int limit=0, max_draft=0;
        std::vector<double> thresholds;
        bool done=true;
        std::string line;
        auto lag=[&]() { return draft ? int(committed.size())-1-draft->past : 0; };
        auto bucket_of=[&](double confidence) {
            int b=0; for(double t:thresholds) if(confidence>=t) ++b; return b;
        };
        while(std::getline(std::cin,line)) {
            try {
                std::istringstream in(line);
                std::string command; in>>command;
                if(command=="QUIT") break;
                if(command=="CONFIG") {
                    int n; in>>max_draft>>n;
                    if(max_draft<1 || max_draft>8 || n<0 || n>8) throw std::runtime_error("Invalid CONFIG");
                    thresholds.assign(n,0.0);
                    for(auto & t:thresholds) in>>t;
                    if(!in) throw std::runtime_error("Invalid CONFIG thresholds");
                    std::cout<<"{\"configured\":true,\"max_draft\":"<<max_draft<<",\"buckets\":"<<thresholds.size()+1<<"}"<<std::endl;
                } else if(command=="RESET") {
                    std::string prompt_hex; in>>limit>>prompt_hex;
                    committed=target.tokenize(unhex(prompt_hex));
                    if(limit<1 || committed.empty() || committed.size()+limit>size_t(context))
                        throw std::runtime_error("Invalid output limit/context");
                    target.clear(); if(draft) draft->clear(); output.clear(); done=false;
                    auto start=Clock::now();
                    Tokens prefix(committed.begin(),committed.end()-1);
                    target.eval(prefix,false); if(draft) draft->eval(prefix,false);
                    std::cout<<"{\"prefill_seconds\":"<<elapsed(start)<<",\"lag\":"<<lag()<<"}"<<std::endl;
                } else if(command=="TEXT") {
                    std::cout<<"{\"text_hex\":\""<<hex(target.text(output))<<"\"}"<<std::endl;
                } else if(command=="PIECES") {
                    // Text of each token id, for showing guesses in the demo dashboard.
                    std::cout<<"{\"pieces_hex\":[";
                    llama_token id; bool first=true;
                    while(in>>id) {
                        if(!first) std::cout<<",";
                        std::cout<<"\""<<hex(target.text(Tokens{id}))<<"\"";
                        first=false;
                    }
                    std::cout<<"]}"<<std::endl;
                } else if(command=="CYCLE") {
                    if(done) throw std::runtime_error("Terminal episode");
                    if(max_draft<1) throw std::runtime_error("Send CONFIG before CYCLE");
                    int draft_first; std::string bits; in>>draft_first>>bits;
                    int buckets=int(thresholds.size())+1;
                    if(bits=="-") bits.clear();
                    if(int(bits.size())!=(max_draft-1)*buckets) throw std::runtime_error("Decision table has wrong size");
                    auto start=Clock::now();
                    int remaining=limit-int(output.size());
                    // Reserve one position for the target's correction or bonus token.
                    int cap=draft ? std::min(max_draft,remaining-1) : 0;
                    int lag_before=lag(), catchup=0;
                    llama_token pending=committed.back();
                    Tokens proposals;
                    std::vector<double> confidences;
                    std::vector<Decision> decisions;
                    double draft_seconds=0;
                    std::vector<double> draft_times;
                    if(draft_first && cap>0) {
                        decisions.push_back({0,-1,1,false});
                        auto t=Clock::now();
                        // Catch-up tokens and the pending token share one batch.
                        int desired=int(committed.size())-1;
                        if(draft->past>desired) draft->rewind(desired);
                        Tokens batch(committed.begin()+draft->past,committed.end());
                        catchup=int(batch.size())-1;
                        draft->eval(batch,false);
                        double confidence;
                        proposals.push_back(draft->greedy_confidence(int(batch.size())-1,confidence));
                        confidences.push_back(confidence);
                        draft_times.push_back(elapsed(t));
                        // Bucket the product of draft confidences: the draft's own
                        // estimate that every token drafted so far is accepted.
                        double prefix=1.0;
                        for(int k=1;;++k) {
                            prefix*=confidences.back();
                            int b=bucket_of(prefix);
                            if(llama_vocab_is_eog(draft->vocab,proposals.back()) || k>=cap) {
                                decisions.push_back({k,b,0,true}); break;
                            }
                            int action=bits[(k-1)*buckets+b]=='1';
                            decisions.push_back({k,b,action,false});
                            if(!action) break;
                            t=Clock::now();
                            draft->eval(Tokens{proposals.back()},true);
                            proposals.push_back(draft->greedy_confidence(0,confidence));
                            confidences.push_back(confidence);
                            draft_times.push_back(elapsed(t));
                        }
                    } else {
                        decisions.push_back({0,-1,0,bool(draft_first)});
                    }
                    for(double v:draft_times) draft_seconds+=v;
                    auto t=Clock::now();
                    int desired=int(committed.size())-1;
                    if(target.past>desired) target.rewind(desired);
                    if(target.past<desired) target.eval(Tokens(committed.begin()+target.past,committed.begin()+desired),false);
                    Tokens verify{pending}; verify.insert(verify.end(),proposals.begin(),proposals.end());
                    target.eval(verify,true);
                    Tokens emitted;
                    int accepted=0; bool mismatch=false;
                    for(size_t i=0;i<proposals.size();++i) {
                        auto token=target.greedy(int(i)); emitted.push_back(token);
                        if(token==proposals[i]) ++accepted;
                        else { mismatch=true; break; }
                        if(llama_vocab_is_eog(target.vocab,token)) break;
                    }
                    if(!mismatch && (emitted.empty() || !llama_vocab_is_eog(target.vocab,emitted.back())))
                        emitted.push_back(target.greedy(int(proposals.size())));
                    double target_seconds=elapsed(t);
                    output.insert(output.end(),emitted.begin(),emitted.end());
                    committed.insert(committed.end(),emitted.begin(),emitted.end());
                    desired=int(committed.size())-1;
                    target.rewind(desired); if(draft) draft->rewind(desired);
                    bool eos=llama_vocab_is_eog(target.vocab,emitted.back());
                    done=eos || int(output.size())>=limit;
                    double seconds=elapsed(start);
                    std::cout<<"{\"drafted\":"<<proposals.size()<<",\"accepted\":"<<accepted
                             <<",\"seconds\":"<<seconds<<",\"draft_seconds\":"<<draft_seconds
                             <<",\"target_seconds\":"<<target_seconds
                             <<",\"lag_before\":"<<lag_before<<",\"catchup\":"<<catchup<<",\"lag_after\":"<<lag()
                             <<",\"done\":"<<(done?"true":"false")<<",\"stop_reason\":";
                    if(done) std::cout<<"\""<<(eos?"eos":"token_limit")<<"\""; else std::cout<<"null";
                    std::cout<<",\"confidences\":[";
                    for(size_t i=0;i<confidences.size();++i) { if(i) std::cout<<","; std::cout<<confidences[i]; }
                    std::cout<<"],\"draft_times\":[";
                    for(size_t i=0;i<draft_times.size();++i) { if(i) std::cout<<","; std::cout<<draft_times[i]; }
                    std::cout<<"],\"decisions\":[";
                    for(size_t i=0;i<decisions.size();++i) {
                        if(i) std::cout<<",";
                        std::cout<<"["<<decisions[i].k<<","<<decisions[i].bucket<<","<<decisions[i].action<<","<<(decisions[i].forced?1:0)<<"]";
                    }
                    std::cout<<"],\"proposals\":[";
                    for(size_t i=0;i<proposals.size();++i) { if(i) std::cout<<","; std::cout<<proposals[i]; }
                    std::cout<<"],\"emitted_ids\":[";
                    for(size_t i=0;i<emitted.size();++i) { if(i) std::cout<<","; std::cout<<emitted[i]; }
                    std::cout<<"]}"<<std::endl;
                } else throw std::runtime_error("Unknown command");
            } catch(const std::exception & e) {
                std::cout<<"{\"error_hex\":\""<<hex(e.what())<<"\"}"<<std::endl;
            }
        }
        return 0;
    } catch(const std::exception & e) {
        std::cout<<"{\"error_hex\":\""<<hex(e.what())<<"\"}"<<std::endl;
        return 1;
    }
}
