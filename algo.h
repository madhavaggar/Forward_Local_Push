//Contributors: Sibo Wang, Renchi Yang
#ifndef __ALGO_H__
#define __ALGO_H__

#include "graph.h"
#include "heap.h"
#include "config.h"
#include "rng.h"
#include <tuple>
#include <boost/random.hpp>
// #include "sfmt/SFMT.h"


struct PredResult{
    double topk_avg_relative_err;
    double topk_avg_abs_err;
    double topk_recall;
    double topk_precision;
    int real_topk_source_count;
    PredResult(double mae=0, double mre=0, double rec=0, double pre=0, int count=0):
        topk_avg_relative_err(mae),
        topk_avg_abs_err(mre),
        topk_recall(rec),
        topk_precision(pre),
        real_topk_source_count(count){}
};

unordered_map<int, PredResult> pred_results;

Fwdidx fwd_idx;
iMap<double> ppr;

// RwIdx rw_idx;
atomic<unsigned long long> num_total_rw;
long num_iter_topk;
vector<int> rw_idx;
vector< pair<unsigned long long, unsigned long> > rw_idx_info;

map< int, vector< pair<int ,double> > > exact_topk_pprs;
vector< pair<int ,double> > topk_pprs;


vector<pair<int, double>> map_lower_bounds;

unsigned concurrency;

vector<int> ks;

inline uint32_t xor128(void){
    static uint32_t x = 123456789;
    static uint32_t y = 362436069;
    static uint32_t z = 521288629;
    static uint32_t w = 88675123;
    uint32_t t;
    t = x ^ (x << 11);   
    x = y; y = z; z = w;   
    return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}

inline static unsigned long new_xshift_lrand(){
    static rng::rng128 rng;
    static uint64_t no_opt=0;
    no_opt |=rng();
    return no_opt;
}

inline static bool new_xshift_drand(){
        return ((double)new_xshift_lrand()/(double)UINT_MAX)< config.alpha;
}

inline static unsigned long xshift_lrand(){
    return (unsigned long)xor128();
}

inline static bool xshift_drand(){
    return ((double)xshift_lrand()/(double)UINT_MAX)<config.alpha;
}

inline static unsigned long lrand() {

    static boost::taus88 rngG(time(0));
    return rngG();

    //return rand();
    // return sfmt_genrand_uint32(&sfmtSeed);
}

inline static bool drand(){
    static boost::bernoulli_distribution <> bernoulli(config.alpha);
    static boost::lagged_fibonacci607 rngG(time(0));
    static boost::variate_generator<boost::lagged_fibonacci607&, boost::bernoulli_distribution<> > bernoulliRngG(rngG, bernoulli);

    return bernoulliRngG();
    //return rand()*1.0f/RAND_MAX;
    // return sfmt_genrand_real1(&sfmtSeed);
}


unsigned int SEED=1;
inline static unsigned long lrand_thd(int core_id) {
    //static thread_local std::mt19937 gen(core_id+1);
    //static std::uniform_int_distribution<> dis(0, INT_MAX);
    //return dis(gen);
    return rand_r(&SEED);
}

inline static double drand_thd(int core_id){
	return ((double)lrand_thd(core_id)/(double)INT_MAX);
}


inline void split_line(){
    INFO("-----------------------------");
}

inline void display_setting(){
    INFO(config.delta);
    INFO(config.pfail);
    INFO(config.rmax);
    INFO(config.omega);
}


static void display_time_usage(int used_counter, int query_size){
   
    if(config.algo == FWDPUSH){
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout <<  Timer::used(FWD_LU)*100.0/Timer::used(used_counter) << "%" << " for forward push cost" << endl;
    }

    if(config.action == TOPK){
        assert(result.real_topk_source_count>0);
        cout << "Average top-K Precision: " << result.topk_precision/result.real_topk_source_count << endl;
        cout << "Average top-K Recall: " << result.topk_recall/result.real_topk_source_count << endl;
    }
    
    cout << "Average query time (s):"<<Timer::used(used_counter)/query_size<<endl;
    cout << "Memory usage (MB):" << get_proc_memory()/1000.0 << endl << endl; 
}

static void set_result(const Graph& graph, int used_counter, int query_size){
    config.query_size = query_size;

    result.m = graph.m;
    result.n = graph.n;
    result.avg_query_time = Timer::used(used_counter)/query_size;

    result.total_mem_usage = get_proc_memory()/1000.0;
    result.total_time_usage = Timer::used(used_counter);

    result.num_randwalk = num_total_rw;
    
    result.randwalk_time = Timer::used(RONDOM_WALK);
    result.randwalk_time_ratio = Timer::used(RONDOM_WALK)*100/Timer::used(used_counter);

    if(config.action == TOPK){
        result.topk_sort_time = Timer::used(SORT_MAP);
        // result.topk_precision = avg_topk_precision;
        // result.topk_sort_time_ratio = Timer::used(SORT_MAP)*100/Timer::used(used_counter);
    }
}

inline void fwdpush_setting(int n, long long m){
    // below is just a estimate value, has no accuracy guarantee
    // since for undirected graph, error |ppr(s, t)-approx_ppr(s, t)| = sum( r(s, v)*ppr(v, t)) <= d(t)*rmax
    // |ppr(s, t)-approx_ppr(s, t)| <= epsilon*ppr(s, t)
    // d(t)*rmax <= epsilon*ppr(s, t)
    // rmax <= epsilon*ppr(s, t)/d(t)
    // d(t): use average degree d=m/n
    // ppr(s, t): use minimum ppr value, delta, i.e., 1/n
    // thus, rmax <= epsilon*delta*n/m = epsilon/m
    // use config.rmax_scale to tune rmax manually
    config.rmax = config.rmax_scale*config.delta*config.epsilon*n/m;
}

inline void generate_ss_query(int n){
    string filename = config.graph_location + "ssquery.txt";
    if(exists_test(filename)){
        INFO("ss query set exists");
        return;
    }
    ofstream queryfile(filename);
    for(int i=0; i<config.query_size; i++){
        int v = rand()%n;
        queryfile<<v<<endl;
    }
}

void load_ss_query(vector<int>& queries){
    string filename = config.graph_location+"ssquery.txt";
     if(!file_exists_test(filename)){
        cerr<<"query file does not exist, please generate ss query files first"<<endl;
        exit(0);
    }
    ifstream queryfile(filename);
    int v;
    while(queryfile>>v){
        queries.push_back(v);
    }
}

void compute_precision(int v){
    double precision=0.0;
    double recall=0.0;
    //INFO(topk_pprs.size());
    if( exact_topk_pprs.size()>=1 && exact_topk_pprs.find(v)!=exact_topk_pprs.end() ){

        unordered_map<int, double> topk_map;
        for(auto &p: topk_pprs){
            if(p.second>0){
                topk_map.insert(p);
            }
        }

        unordered_map<int, double> exact_map;
        int size_e = min( config.k, (unsigned int)exact_topk_pprs[v].size() );

        for(int i=0; i<size_e; i++){
            pair<int ,double>& p = exact_topk_pprs[v][i];
            if(p.second>0){
                exact_map.insert(p);
                if(topk_map.find(p.first)!=topk_map.end())
                    recall++;
            }
        }

        for(auto &p: topk_map){
            if(exact_map.find(p.first)!=exact_map.end()){
                precision++;
            }
        }

        assert(exact_map.size() > 0);
        assert(topk_map.size() > 0);


        recall = recall*1.0/exact_map.size();
        precision = precision*1.0/exact_map.size();
        INFO(exact_map.size(), recall, precision);
        result.topk_recall += recall;
        result.topk_precision += precision;

        result.real_topk_source_count++;
    }
}

inline bool cmp(double x, double y){
    return x>y;
}

double topk_ppr(){
    topk_pprs.clear();
    topk_pprs.resize(config.k);

    static unordered_map< int, double > temp_ppr;
    temp_ppr.clear();
    // temp_ppr.resize(ppr.occur.m_num);
    int nodeid;
    for(long i=0; i<ppr.occur.m_num; i++){
        nodeid = ppr.occur[i];
        // INFO(nodeid);
        temp_ppr[nodeid] = ppr[ nodeid ];
    }

    partial_sort_copy(temp_ppr.begin(), temp_ppr.end(), topk_pprs.begin(), topk_pprs.end(), 
            [](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});
    
    return topk_pprs[config.k-1].second;
}

void compute_precision_for_dif_k(int v){
    if( exact_topk_pprs.size()>=1 && exact_topk_pprs.find(v)!=exact_topk_pprs.end() ){
        for(auto k: ks){

            int j=0;
            unordered_map<int, double> topk_map;
            for(auto &p: topk_pprs){
                if(p.second>0){
                    topk_map.insert(p);
                }
                j++;
                if(j==k){ // only pick topk
                    break;
                }
            }

            double recall=0.0;
            unordered_map<int, double> exact_map;
            int size_e = min( k, (int)exact_topk_pprs[v].size() );
            for(int i=0; i<size_e; i++){
                pair<int ,double>& p = exact_topk_pprs[v][i];
                if(p.second>0){
                    exact_map.insert(p);
                    if(topk_map.find(p.first)!=topk_map.end())
                        recall++;
                }
            }

            double precision=0.0;
            for(auto &p: topk_map){
                if(exact_map.find(p.first)!=exact_map.end()){
                    precision++;
                }
            }

            //if(exact_map.size()<=1)
            //   continue;

            precision = precision*1.0/exact_map.size();
            recall = recall*1.0/exact_map.size();

            pred_results[k].topk_precision += precision;
            pred_results[k].topk_recall += recall;
            pred_results[k].real_topk_source_count++;
        }
    }
}

inline void display_precision_for_dif_k(){
    split_line();
    cout << config.algo << endl;
    for(auto k: ks){
        cout << k << "\t";
    }
    cout << endl << "Precision:" << endl;
    //assert(pred_results[k].real_topk_source_count>0);
    for(auto k: ks){
        cout << pred_results[k].topk_precision/pred_results[k].real_topk_source_count << "\t";
    }
    cout << endl << "Recall:" << endl;
    for(auto k: ks){
        cout << pred_results[k].topk_recall/pred_results[k].real_topk_source_count << "\t";
    }
    cout << endl;
}

void forward_local_update_linear(int s, const Graph &graph, double& rsum, double rmax, double init_residual = 1.0){
    fwd_idx.first.clean();
    fwd_idx.second.clean();

    static vector<bool> idx(graph.n);
    std::fill(idx.begin(), idx.end(), false);

    if(graph.g[s].size()==0){
        fwd_idx.first.insert( s, 1);
        rsum =0;
        return; 
    }

    double myeps = rmax;//config.rmax;

    vector<int> q;  //nodes that can still propagate forward
    q.reserve(graph.n);
    q.push_back(-1);
    unsigned long left = 1;
    q.push_back(s);

    // residual[s] = init_residual;
    fwd_idx.second.insert(s, init_residual);
    
    idx[s] = true;
    
    while (left < (int) q.size()) {
        int v = q[left];
        idx[v] = false;
        left++;
        double v_residue = fwd_idx.second[v];
        fwd_idx.second[v] = 0;
        if(!fwd_idx.first.exist(v))
            fwd_idx.first.insert( v, v_residue * config.alpha);
        else
            fwd_idx.first[v] += v_residue * config.alpha;

        int out_neighbor = graph.g[v].size();
        rsum -=v_residue*config.alpha;
        if(out_neighbor == 0){
            fwd_idx.second[s] += v_residue * (1-config.alpha);
            if(graph.g[s].size()>0 && fwd_idx.second[s]/graph.g[s].size() >= myeps && idx[s] != true){
                idx[s] = true;
                q.push_back(s);   
            }
            continue;
        }

        double avg_push_residual = ((1.0 - config.alpha) * v_residue) / out_neighbor;
        for (int next : graph.g[v]) {
            // total_push++;
            if( !fwd_idx.second.exist(next) )
                fwd_idx.second.insert( next,  avg_push_residual);
            else
                fwd_idx.second[next] += avg_push_residual;

            //if a node's' current residual is small, but next time it got a large residual, it can still be added into forward list
            //so this is correct
            if ( fwd_idx.second[next]/graph.g[next].size() >= myeps && idx[next] != true) {  
                idx[next] = true;//(int) q.size();
                q.push_back(next);    
            }
        }
    }
}

extern double threshold;

inline double calculate_lambda(double rsum, double pfail, double upper_bound, long total_rw_num){
    return 1.0/3*log(2/pfail)*rsum/total_rw_num + 
    sqrt(4.0/9.0*log(2.0/pfail)*log(2.0/pfail)*rsum*rsum +
        8*total_rw_num*log(2.0/pfail)*rsum*upper_bound)
    /2.0/total_rw_num;
}

double threshold = 0.0;

#endif
