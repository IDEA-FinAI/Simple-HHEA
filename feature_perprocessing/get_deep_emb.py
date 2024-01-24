import json
import argparse
import gensim
import torch
import numpy as np
from preproccess import load_ents


def get_node_vec(wv, node2same, ents):
    node_vec={}
    for id in ents:
        new_id = id
        if id in node2same:
            new_id = node2same[id]
        try:
            node_vec[id] = wv[new_id]   # dim=64
        except:
            node_vec[id] = np.random.random_sample(size = (1, 64))[0]
            continue
    
    vec = []
    for i in range(len(ents)):
        vec.append(node_vec[str(i)])
    vec = torch.tensor(np.array(vec), requires_grad=False)
    print(vec.shape)
    return vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,default="data/icews_wiki/")
    args = parser.parse_args()
    path = args.path

    ### load entities
    ent_1 = load_ents(path+'id_features_1')
    print('load ent_1',len(ent_1))
    ent_2 = load_ents(path+'id_features_2')
    print('load ent_2',len(ent_2))
    ents = ent_1.copy()
    ents.update(ent_2)
    print('load ents',len(ents))

    with open(path + "node2same", "r") as fr:
        node2same = json.load(fr)
    print("node2same:", len(node2same))

    ### get deepwalk embeddings
    deep_wv = gensim.models.KeyedVectors.load_word2vec_format(path + "longterm.vec")
    deep_node = get_node_vec(deep_wv, node2same, ents)
    deep_emb = deep_node.numpy()

    ### save embeddings
    np.savetxt(path + "deep_emb.txt", deep_emb)
