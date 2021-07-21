
# Single Source Personalized PageRank

## Tested Environment
- Ubuntu
- C++ 11
- GCC 4.8
- Boost (Boost 1.65 on our machine)
- cmake

## Compile
```sh
$ cmake .
$ make
```

## Parameters
```sh
./fora action_name --algo <algorithm> [options]
```
- action:
    - query
    - topk
    - generate-ss-query: generate queries file
    - gen-exact-topk: generate ground truth by power-iterations method
    - batch-topk: for different k=i*config.k/5, i=1,2,3,4,5, compute precision
- algo: which algorithm you prefer to run
    - fwdpush: Forward Push
    - revpush: Reverse Push 
- options
    - --prefix \<prefix\>
    - --epsilon \<epsilon\>
    - --dataset \<dataset\>
    - --query_size \<queries count\>
    - --k \<top k\>
    - --exact_ppr_path \<directory to place generated ground truth\>
    - --result_dir \<directory to place results\>
    - --balanced: a balance strategy is used to automatically decide R_max for FORA.
    - --opt:  optimization techniques for whole-graph SSPPR and top-k queries are applied.

## Data
The example data format is in `./data/webstanford/` folder. The data for DBLP, Pokec, Livejournal, Twitter are not included here for size limitation reason. You can find them online. For datasets with node numbers greater than the node count kindly reassign `remap` variable in config.h to True.

config.remap = True;

```sh
make

```

## Generate queries
Generate query files for the graph data. Each line contains a node id.

```sh
$ ./fora generate-ss-query --prefix <data-folder> --dataset <graph-name> --query_size <query count>
```

- Example:

```sh
$ ./fora generate-ss-query --prefix ./data/ --dataset webstanford --query_size 1000
```


## Query
Process queries.

```sh
$ ./fora query --algo <algo-name> --prefix <data-folder> --dataset <graph-name> --result_dir <output-folder> --epsilon <relative error> --query_size <query count> (--opt)
```

- Example:

```sh

$ ./fora query --algo fwdpush --prefix ./data/ --dataset webstanford --epsilon 0.5 --query_size 20



## Top-K
Process top-k queries.

```sh
$ ./fora topk --algo <algo-name> --prefix <data-folder> --dataset <graph-name> --result_dir <output-folder> --epsilon <relative error> --query_size <query count> --k <k>
```

- Example

```sh

$ ./fora topk --algo fwdpush --prefix ./data/ --dataset webstanford --epsilon 0.5 --query_size 20 --k 500


```


## Exact PPR (ground truth)
Construct ground truth for the generated queries.

```sh
$ ./fora gen-exact-topk --prefix <data-folder> --dataset <graph-name> --k <k> --query_size <query count> --exact_ppr_path <folder to save exact ppr>
```

- Example

```sh
$ mkdir ./exact
$ ./fora gen-exact-topk --prefix ./data/ --dataset webstanford --k 1000 --query_size 100 --exact_ppr_path ./exact/
```


## Batch Top-K
Specify k value, and process queries for various k values (k/5, 2k/5, 3k/5, 4k/5, k), the folder contains ground truth need to be specified to obtain precision.

```sh
$ ./fora batch-topk --algo <algo-name> --prefix <data-folder> --dataset <graph-name> --result_dir <output-folder> --epsilon <relative error> --query_size <query count> --k <k> --exact_ppr_path <folder contains ground truth>
```

- Example

```sh
$ ./fora batch-topk --algo fwdpush --prefix ./data/ --dataset webstanford --epsilon 0.5 --query_size 20 --k 500 --exact_ppr_path ./exact/
```
