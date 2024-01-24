# Simple-HHEA

The code and dataset for papar ***Toward Practical Entity Alignment Method Design: Insights from New Highly Heterogeneous Knowledge Graph Datasets*** in The Web Conf 2024.



## Environment

```
Python
Pytorch
transformers
SentencePiece
scipy
numpy
pandas
tqdm
networkx
gensim
```



## How to Run

The model runs in 3 steps:

#### 1. Get the name embeddings

To get the name embeddings of entities, use:

```bash
python process_name_embedding.py --data DATASET
```

`DATASET` can be `icews_wiki`, `icews_yago` or any dataset you place in the directory [data](./data).

#### 2. Get the structure embeddings

We use [Fualign](https://github.com/showerage/fualign) to get the embeddings of entities by deepwalk. To get the structure embeddings, use: 

```bash
cd fualign
python preprocess.py --l DATASET
python longterm/main.py \
	--input "data/DATASET/deepwalk.data" \
	--output "data/DATASET/longterm.vec" \
	--node2rel "data/DATASET/node2rel" \
	--q 0.7
python get_deep_emb.py --path "data/DATASET/"
```

`DATASET` is the same as the one in **Step 1**.

#### 3. Run Simple-HHEA

To run Simple-HHEA, use:

```bash
python main_SimpleHHEA.py \
	--data DATASET \
	--lr 0.01 \
    --wd 0.001 \
    --gamma 1.0 \
    --epochs 1500
```

use `--add_noise` and `--noise_ratio` to control whether to add noise to the name embeddings and how much noise.

use `--no_structure` to remove structure embeddings from model.

use`--no_time` to remove time embeddings from model.



Or you can use:

```bash
bash run_exp.sh
```

to directly run Simple-HHEA on dataset icews_wiki.

