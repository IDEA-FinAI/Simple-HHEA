#!/bin/bash
cuda=3
lr=0.01
wd=0.001
gamma=1.0
epochs=1500

L=("icews_wiki")
for lang in ${L[*]}
do
    echo "### get name embeddings for $lang"
    python process_name_embedding.py --data $lang

    echo "### get deepwalk embeddings: preprocess for $lang"
    python feature_perprocessing/preprocess.py --l $lang
    echo "### get deepwalk embeddings: get longterm.vec for $lang"
    python feature_perprocessing/longterm/main.py --input "data/$lang/deepwalk.data" --output "data/$lang/longterm.vec" --node2rel "data/$lang/node2rel" --q 0.7
    echo "### get deepwalk embeddings: save deepwalk embeddings for $lang"
    python feature_perprocessing/get_deep_emb.py --path "data/$lang/"

    echo "run main experiment for $lang"
    python main_SimpleHHEA.py --data $lang --cuda $cuda --lr $lr --wd $wd --gamma $gamma --epochs $epochs
done