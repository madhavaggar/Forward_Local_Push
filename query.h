//Contributors: Sibo Wang, Renchi Yang
#ifndef FORA_QUERY_H
#define FORA_QUERY_H
//#define ska::bytell_hash_map ska::ska::bytell_hash_map

#include "algo.h"
#include "graph.h"
#include "heap.h"
#include "config.h"
#include "build.h"
#include <omp.h>
#include <thread>
#include <chrono>

//#define CHECK_PPR_VALUES 1
// #define CHECK_TOP_K_PPR 1
#define PRINT_PRECISION_FOR_DIF_K 1
#define NUM_THREADS 64
// std::mutex mtx;


using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


void compute_ppr_with_reserve(Fwdidx &fwd_idx, iMap<double> &ppr){
    ppr.clean();
    int node_id;
    double reserve;
    //printf("number of nodes that are touched is %d\n", fwd_idx.first.occur.m_num);
    for(long i=0; i< fwd_idx.first.occur.m_num; i++){
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[ node_id ];
        //printf("%d, %f\n", node_id, reserve);
        if(reserve)
            ppr.insert(node_id, reserve);
    }
}

void compute_ppr_with_reserve_reverse(Bwdidx &bwd_idx, iMap<double> &ppr){
    ppr.clean();
    int node_id;
    double reserve;
    //printf("number of nodes that are touched is %d\n", bwd_idx.first.occur.m_num);
    for(long i=0; i< bwd_idx.first.occur.m_num; i++){
        node_id = bwd_idx.first.occur[i];
        reserve = bwd_idx.first[ node_id ];
        //printf("%d, %f\n", node_id, reserve);
        if(reserve)
            ppr.insert(node_id, reserve);
    }
}

void get_topk_fwdpush(int v, Graph &graph, Fwdidx &fwd_idx, iMap<double> &ppr){ // 1 thread 1 query
    //display_setting();

    Timer timer(0);
    double rsum = 1;

    {
        Timer timer(FWD_LU);
        forward_local_update_linear(v, graph, rsum, config.rmax, fwd_idx);
        //printf("config.rmax is  %.9f\n",  config.rmax);
    }
    
    compute_ppr_with_reserve(fwd_idx, ppr);
    vector< pair<int ,double> > topk_pprs;
    topk_ppr(ppr, topk_pprs);
    // not FORA, so it's single source
    // no need to change k to un again
    // check top-k results for different k
    
    compute_precision_for_dif_k(v,topk_pprs);

    compute_precision(v, topk_pprs);

#ifdef CHECK_TOP_K_PPR
    vector<pair<int, double>>& exact_result = exact_topk_pprs[v];
    INFO("query node:", v);
    for(int i=0; i<topk_pprs.size(); i++){
        cout << "Estimated k-th node: " << topk_pprs[i].first << " PPR score: " << topk_pprs[i].second << " " << map_lower_bounds[topk_pprs[i].first].first<< " " << map_lower_bounds[topk_pprs[i].first].second
             <<" Exact k-th node: " << exact_result[i].first << " PPR score: " << exact_result[i].second << endl;
    }
#endif
}

void get_topk_revpush(int v, Graph &graph, Bwdidx &bwd_idx, iMap<double> &ppr){ // 1 thread 1 query
    //display_setting();

    Timer timer(0);
    double rsum = 1;

    {
        Timer timer(FWD_LU);
        reverse_local_update_linear(v, graph, bwd_idx);
    }
    compute_ppr_with_reserve_reverse(bwd_idx, ppr);
    vector< pair<int ,double> > topk_pprs;
    topk_ppr(ppr, topk_pprs);
    // not FORA, so it's single source
    // no need to change k to un again
    // check top-k results for different k
    compute_precision_for_dif_k(v,topk_pprs);

    compute_precision(v, topk_pprs);

#ifdef CHECK_TOP_K_PPR
    vector<pair<int, double>>& exact_result = exact_topk_pprs[v];
    INFO("query node:", v);
    for(int i=0; i<topk_pprs.size(); i++){
        cout << "Estimated k-th node: " << topk_pprs[i].first << " PPR score: " << topk_pprs[i].second << " " << map_lower_bounds[topk_pprs[i].first].first<< " " << map_lower_bounds[topk_pprs[i].first].second
             <<" Exact k-th node: " << exact_result[i].first << " PPR score: " << exact_result[i].second << endl;
    }
#endif
}


void fwd_power_iteration(const Graph& graph, int start, ska::bytell_hash_map<int, double>& map_ppr){
    static thread_local ska::bytell_hash_map<int, double> map_residual;
    map_residual[start] = 1.0;

    int num_iter=0;
    double rsum = 1.0;
    while( num_iter < config.max_iter_num ){
        num_iter++;
        // INFO(num_iter, rsum);
        vector< pair<int,double> > pairs(map_residual.begin(), map_residual.end());
        map_residual.clear();
        for(const auto &p: pairs){
            if(p.second > 0){
                map_ppr[p.first] += config.alpha*p.second;
                int out_deg = graph.g[p.first].size();

                double remain_residual = (1-config.alpha)*p.second;
                rsum -= config.alpha*p.second;
                if(out_deg==0){
                    map_residual[start] += remain_residual;
                }
                else{
                    double avg_push_residual = remain_residual / out_deg;
                    for (int next : graph.g[p.first]) {
                        map_residual[next] += avg_push_residual;
                    }
                }
            }
        }
        pairs.clear();
    }
    map_residual.clear();
}

void multi_power_iter(const Graph& graph, const vector<int>& source, ska::bytell_hash_map<int, vector<pair<int ,double>>>& map_topk_ppr ){
    static thread_local ska::bytell_hash_map<int, double> map_ppr;
    for(int start: source){
        fwd_power_iteration(graph, start, map_ppr);

        vector<pair<int ,double>> temp_top_ppr(config.k);
        partial_sort_copy(map_ppr.begin(), map_ppr.end(), temp_top_ppr.begin(), temp_top_ppr.end(), 
            [](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});
        
        map_ppr.clear();
        map_topk_ppr[start] = temp_top_ppr;
    }
}

void gen_exact_topk(const Graph& graph){

    vector<long long> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned long long query_size = queries.size();
    query_size = min( query_size, config.query_size );
    INFO(query_size);
    string exact_top_file_str = get_exact_topk_ppr_file();
    if(exists_test(exact_top_file_str)){
        INFO("exact top k exists");
        return;
    }
    assert(config.k < graph.n-1);
    assert(config.k > 1);
    INFO(config.k);

    split_line();

    unsigned long long NUM_CORES = std::thread::hardware_concurrency()-1;
    assert(NUM_CORES >= 2);

    int num_thread = min(query_size, NUM_CORES);
    int avg_queries_per_thread = query_size/num_thread;

    vector<vector<int>> source_for_all_core(num_thread);
    vector<ska::bytell_hash_map<int, vector<pair<int ,double>>>> ppv_for_all_core(num_thread);

    for(int tid=0; tid<num_thread; tid++){
        int s = tid*avg_queries_per_thread;
        int t = s+avg_queries_per_thread;

        if(tid==num_thread-1)
            t+=query_size%num_thread;

        for(;s<t;s++){
            // cout << s+1 <<". source node:" << queries[s] << endl;
            source_for_all_core[tid].push_back(queries[s]);
        }
    }


    {
        Timer timer(PI_QUERY);
        INFO("power itrating...");
        std::vector< std::future<void> > futures(num_thread);
        for(int tid=0; tid<num_thread; tid++){
            futures[tid] = std::async( std::launch::async, multi_power_iter, std::ref(graph), std::ref(source_for_all_core[tid]), std::ref(ppv_for_all_core[tid]) );
        }
        std::for_each( futures.begin(), futures.end(), std::mem_fn(&std::future<void>::wait));
    }

    // cout << "average iter times:" << num_iter_topk/query_size << endl;
    cout << "average generation time (s): " << Timer::used(PI_QUERY)*1.0/query_size << endl;

    INFO("combine results...");
    for(int tid=0; tid<num_thread; tid++){
        for(auto &ppv: ppv_for_all_core[tid]){
            exact_topk_pprs.insert( ppv );
        }
        ppv_for_all_core[tid].clear();
    }

    save_exact_topk_ppr();
}

void topk(Graph& graph){
    vector<long long> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned long long query_size = queries.size();
    query_size = min( query_size, config.query_size );
    int used_counter;

    assert(config.k < graph.n-1);
    assert(config.k > 1);
    INFO(config.k);
    
    
    printf("The alpha config.alpha is %.9f\n", config.alpha);

    split_line();

    // not FORA, so it's single source
    // no need to change k to run again
    // check top-k results for different k
    unsigned int step = config.k / 5;
    if (step > 0)
    {
        for (unsigned int i = 1; i < 5; i++)
        {
            ks.push_back(i * step);
        }
    }
    ks.push_back(config.k);
    for (auto k : ks)
    {
        PredResult rst(0, 0, 0, 0, 0);
        pred_results.insert(MP(k, rst));
    }

    used_counter = 0; 
  
    load_exact_topk_ppr();
    omp_set_num_threads(NUM_THREADS);
    //double t1 = omp_get_wtime();
    
    auto t1 = high_resolution_clock::now();
    
    if(config.algo == FWDPUSH){
        
        #pragma omp parallel for
        for(int i=0; i<query_size; i++){ //1000 source nodes
            
            used_counter=0;
            iMap<double> * ppr = new iMap<double>;
            fwdpush_setting(graph.n, graph.m);
            (*ppr).initialize(graph.n);
            Fwdidx * fwd_idx = new Fwdidx;
            (*fwd_idx).first.initialize(graph.n);
            (*fwd_idx).second.initialize(graph.n);
            //cout << i + 1 << ". source node:" << queries[i] << endl;
            get_topk_fwdpush(queries[i], graph, * fwd_idx, * ppr);
            
            (*fwd_idx).first.free_mem();
            (*fwd_idx).second.free_mem();
            ppr->free_mem();
            
            delete ppr;
            delete fwd_idx;
            
            
            //split_line();
        }
        
    }
    else if(config.algo == REVPUSH){
        #pragma omp parallel for
        for (int i = 0; i < query_size; i++){ //1000 source nodes
            used_counter=0;
            Bwdidx * bwd_idx = new Bwdidx;
            iMap<double> * ppr = new iMap<double>;
            fwdpush_setting(graph.n, graph.m);
            (*ppr).initialize(graph.n);
            (*bwd_idx).first.initialize(graph.n);
            (*bwd_idx).second.initialize(graph.n);
            //cout << i + 1 << ". source node:" << queries[i] << endl;
            get_topk_revpush(queries[i], graph, * bwd_idx, * ppr);
            
            (*bwd_idx).first.free_mem();
            (*bwd_idx).second.free_mem();
            ppr->free_mem();
            //split_line();
            delete ppr;
            delete bwd_idx;
        }
    }


    //double t2 = omp_get_wtime();
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    
    std::cout << "TIME FOR PPR: " << ms_double.count() << "ms\n";
    
    //cout << "TIME FOR PPR: " << t2 - t1 << endl;
    //cout<<"Query Time:"<<query_size<<endl;
    cout << "average iter times:" << ms_double.count()/query_size << "ms\n";
    display_time_usage(used_counter, query_size);
    set_result(graph, used_counter, query_size);

    // not FORA, so it's single source
    // no need to change k to run again
    // check top-k results for different k
    display_precision_for_dif_k();

}




void querythread(int start_add, int end_add, vector<long long> & queries, Graph& graph, Fwdidx & fwd_idx, double rsum, iMap<double> & ppr){


    fwd_idx.first.clean();
    fwd_idx.second.clean();
    ppr.clean();

    //std::cout << "start and end is " <<  start_add << " "<< end_add << endl;
    //std::thread::id this_id = std::this_thread::get_id();
    //std::cout << "thread " << this_id << endl;

    for(int iq = start_add; iq < end_add; iq++){
        forward_local_update_linear(queries[iq], graph, rsum, config.rmax, fwd_idx);
        //     //cout<<"After forward_local_update"<<endl;
        compute_ppr_with_reserve(fwd_idx, ppr);
        fwd_idx.first.clean();
        fwd_idx.second.clean();
        ppr.clean();
    }


}



void query(Graph& graph){
    INFO(config.algo);
    vector<long long> queries;
    load_ss_query(queries);
    unsigned long long query_size = queries.size();
    query_size = min( query_size, config.query_size );
    INFO(query_size);
    int used_counter;

    assert(config.rmax_scale >= 0);
    INFO(config.rmax_scale);

    omp_set_num_threads(NUM_THREADS);  


    std::vector<std::thread> threads;


    int numberOfqueryPerThread = query_size/NUM_THREADS;
       
    auto t1 = high_resolution_clock::now();
    if(config.algo == FWDPUSH){
        used_counter=0;
        fwdpush_setting(graph.n, graph.m);
        double rsum = 1;
        used_counter = FWD_LU;
        Timer timer(used_counter);
        iMap<double> ppr[NUM_THREADS];
        Fwdidx fwd_idx[NUM_THREADS];
        for(int it = 0; it < NUM_THREADS; it++){
            ppr[it].init_keys(graph.n);
            fwd_idx[it].first.initialize(graph.n);
            fwd_idx[it].second.initialize(graph.n);
        }

        for(int it = 0; it < NUM_THREADS; it++){
            printf("allocate thread\n");
            if(it < NUM_THREADS  - 1){
                threads.push_back(std::thread(
                    querythread,
                    it * numberOfqueryPerThread,
                    (it + 1) * numberOfqueryPerThread,
                    std::ref(queries),
                    std::ref(graph),
                    std::ref(fwd_idx[it]),
                    rsum,
                    std::ref(ppr[it])
                    ));
            }
            else{
                threads.push_back(std::thread(
                    querythread, 
                    it * numberOfqueryPerThread,
                    query_size,
                    std::ref(queries),
                    std::ref(graph),
                    std::ref(fwd_idx[it]),
                    rsum,
                    std::ref(ppr[it])
                    ));
            }
        }

        


        for(int it = 0; it < NUM_THREADS; it++){
            threads[it].join();
            ppr[it].free_mem();
            fwd_idx[it].first.free_mem();
            fwd_idx[it].second.free_mem();
        }
        





        // #pragma omp parallel for
        // for(long long i=0; i<query_size; i++){ //parallelize pardo using multiple threads: multiple sources at once
        //     used_counter=0;
        //     iMap<double> ppr;
        //     ppr.init_keys(graph.n);
        //     fwdpush_setting(graph.n, graph.m);
        //     //display_setting();
        //     used_counter = FWD_LU;
        //     Fwdidx fwd_idx;
        //     fwd_idx.first.initialize(graph.n);
        //     fwd_idx.second.initialize(graph.n);
        //     //cout << i+1 <<". source node:" << queries[i] << endl;
        //     Timer timer(used_counter);
        //     double rsum = 1;
        //     //cout<<"Before forward_local_update"<<endl;
        //     forward_local_update_linear(queries[i], graph, rsum, config.rmax, fwd_idx);
        //     //cout<<"After forward_local_update"<<endl;
        //     compute_ppr_with_reserve(fwd_idx, ppr);
        //     fwd_idx.first.free_mem();
        //     fwd_idx.second.free_mem();
        //     ppr.free_mem();
            
        //     //cout<<"Afterreserve computation"<<endl;
        //     //split_line();
        // }
    }
    else if(config.algo == REVPUSH){
        
        #pragma omp parallel for
        for(int i=0; i<query_size; i++){
            used_counter=0;
            iMap<double> ppr;
            ppr.init_keys(graph.n);
            fwdpush_setting(graph.n, graph.m);
            //display_setting();
            used_counter = FWD_LU;
            Bwdidx bwd_idx; 
            bwd_idx.first.initialize(graph.n);
            bwd_idx.second.initialize(graph.n);
            
            //cout << i+1 <<". source node:" << queries[i] << endl;
            Timer timer(used_counter);
            double rsum = 1;
            reverse_local_update_linear(queries[i], graph, bwd_idx);
            compute_ppr_with_reserve_reverse(bwd_idx, ppr);
            //split_line();
            bwd_idx.first.free_mem();
            bwd_idx.second.free_mem();
            ppr.free_mem();
        }

    }
     auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    
    std::cout << "TIME FOR PPR: " << ms_double.count() << "ms\n";
    
    //cout << "TIME FOR PPR: " << t2 - t1 << endl;
    //cout<<"Query Time:"<<query_size<<endl;
    cout << "average iter times:" << ms_double.count()/query_size << "ms\n";
    
    display_time_usage(used_counter, query_size);
    set_result(graph, used_counter, query_size);
}

void batch_topk(Graph& graph){
    vector<long long> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned long long query_size = queries.size();
    query_size = min( query_size, config.query_size );
    int used_counter=0;
    assert(config.k < graph.n-1);
    assert(config.k > 1);
    INFO(config.k);

    split_line();
    load_exact_topk_ppr();

    used_counter = 0;

    unsigned int step = config.k/5;
    if(step > 0){
        for(unsigned int i=1; i<5; i++){
            ks.push_back(i*step);
        }
    }
    ks.push_back(config.k);
    for(auto k: ks){
        PredResult rst(0,0,0,0,0);
        pred_results.insert(MP(k, rst));
    }

    // not FORA, so it's of single source
    // no need to change k to run again
    // check top-k results for different k

    omp_set_num_threads(NUM_THREADS);

    if(config.algo == FWDPUSH){

        #pragma omp parallel for
        for (int i = 0; i < query_size; i++)
        {
            Fwdidx fwd_idx;
            iMap<double> ppr;
            fwdpush_setting(graph.n, graph.m);
            ppr.initialize(graph.n);

            fwd_idx.first.initialize(graph.n);
            fwd_idx.second.initialize(graph.n);

            //cout << i + 1 << ". source node:" << queries[i] << endl;
            get_topk_fwdpush(queries[i], graph, fwd_idx, ppr);

            //split_line();
        }
    }

    else if(config.algo == REVPUSH){

        #pragma omp parallel for
        for (int i = 0; i < query_size; i++)
        {
            Bwdidx bwd_idx;
            iMap<double> ppr;
            fwdpush_setting(graph.n, graph.m);
            ppr.initialize(graph.n);

            bwd_idx.first.initialize(graph.n);
            bwd_idx.second.initialize(graph.n);

            //cout << i + 1 << ". source node:" << queries[i] << endl;
            get_topk_revpush(queries[i], graph, bwd_idx, ppr);



            //split_line();
        }
    }

    display_time_usage(used_counter, query_size);
    set_result(graph, used_counter, query_size);

    display_precision_for_dif_k();
}

#endif 
