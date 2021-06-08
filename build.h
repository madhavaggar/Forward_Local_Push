#ifndef BUILD_H
#define BUILD_H

#include "algo.h"
#include "graph.h"
#include "heap.h"
#include "config.h"

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>
#include <sys/sysinfo.h>

inline size_t get_ram_size(){
    struct sysinfo si;
    sysinfo (&si);
    return si.totalram;
}

inline string get_exact_topk_ppr_file(){
    if(!boost::algorithm::ends_with(config.exact_pprs_folder, FILESEP))
        config.exact_pprs_folder += FILESEP;
    return config.exact_pprs_folder+config.graph_alias+".topk.pprs";
}

inline void save_exact_topk_ppr(){
    string filename = get_exact_topk_ppr_file();
    std::ofstream ofs(filename);
    boost::archive::text_oarchive oa(ofs);
    oa << exact_topk_pprs;
}

inline void load_exact_topk_ppr(){
    string filename = get_exact_topk_ppr_file();
    if(!exists_test(filename)){
        INFO("No exact topk ppr file", filename);
        return;
    }
    std::ifstream ifs(filename);
    boost::archive::text_iarchive ia(ifs);
    ia >> exact_topk_pprs;

    INFO(exact_topk_pprs.size());
}



#endif
