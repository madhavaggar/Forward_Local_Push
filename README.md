
# Single Source Personalized PageRank

## Tested Environment
- Linux Ubuntu 
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
```sh `
./fora action_name --algo <algorithm> [options]
```
- action:
    - query
    - topk
    - generate-ss-query: generate queries file
    - gen-exact-topk: generate ground truth by power-iterations method
- algo: which algorithm you prefer to run
    - fwdpush: Forward Push
    - revpush: Reverse Push 
- options
    - --prefix \<prefix\>
    - --epsilon \<epsilon\>
    - --dataset \<dataset\> [Can use any dataset]
    - --query_size \<queries count\>
    - --k \<top k\>
    - --exact_ppr_path \<directory to place generated ground truth\>
    - --result_dir \<directory to place results\>

## Data
The example data format is in `./data/webstanford/` folder. The data for DBLP, Pokec, Livejournal, Twitter, Friendster are not included here for size limitation reason. You can find them online. For datasets with **node numbers greater than the node count** kindly reassign `remap` variable in config.h to True and define maxid in attribute.txt and then recompile. Set the parameter after `--dataset` in order to change the dataset. You can download the datasets at https://snap.stanford.edu/data/. Define your attribute.txt under `./data/<dataset>` along with graph.txt as given below.

```
m = (number of nodes), 
n = (number of edges), 
maxid = (maximum possible node ID)

```

```sh
make

```

## Changing the Number of Threads
In order to change the number of threads you can set the `NUM_THREADS` macro under query.h to a number of your choice. Post the change please run:

```shID
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
