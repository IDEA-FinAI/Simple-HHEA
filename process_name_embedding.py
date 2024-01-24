import os
import argparse
import torch
from tqdm import tqdm
from transformers import AlbertTokenizer
from transformers import AlbertModel
from transformers import logging
logging.set_verbosity_warning()
import numpy as np
import pandas as pd


class BertEmbedding:
    def __init__(self):
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        self.tokenizer = tokenizer
        self.model = AlbertModel.from_pretrained("albert-base-v2").cuda()

    def embed_new(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
        outputs = torch.mean(self.model(**inputs).last_hidden_state, dim=1)
        return outputs[0].detach()

def process_id_features(data_dir, file_name, output_name):
    id_features = []
    with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            eid, name = line.strip().split("\t")
            name = name.split("/")[-1].replace("_", " ").replace(u'\xa0', '')
            id_features.append((eid, name))
    with open(os.path.join(data_dir, output_name), "w", encoding="utf-8") as fw:
        for eid, name in id_features:
            fw.write(f"{eid}\t{name}\n")

def init_embedding_encoder(data_dir, file_name):
    extractor = BertEmbedding()
    emb_file = open(os.path.join(data_dir, file_name + "_emb"), "w+", encoding = "utf-8")
    with open(os.path.join(data_dir, file_name), "r", encoding = "utf-8") as fr:
        for line in tqdm(fr.readlines(), desc=f"{file_name} init"):
            line = line.replace("\n", "").split("\t")
            ent_id = line[0]
            ent_name = line[1]
            ent_emb = np.array(now_extractor(ent_name, extractor)) #生成768维的embedding
            str_ent_emb = ""
            ent_emb_list = [("%.8f" % i) for i in ent_emb.tolist()]
            for item in ent_emb_list:
                str_ent_emb += (str(item) + ",")
            str_ent_emb = str_ent_emb[: -1]
            final_str = ent_id + "\t" + str_ent_emb + "\n"
            emb_file.write(final_str)
    emb_file.close()


def now_extractor(text, extractor):
    # 进行字符串预处理，以及特征生成
    ent_name = (text).replace("\t", "").replace("_", " ").replace("/", " ")
    ent_emb = np.array(extractor.embed_new(ent_name).cpu())
    news_embed = ent_emb.astype(np.float32)
    return news_embed

def compute_kernel_bias(vecs, n_components=256):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """ 最终向量标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="icews_wiki")
    args = parser.parse_args()
    data_dir = os.path.join("data", args.data)

    if not os.path.exists(os.path.join(data_dir, "ent_ids_1_emb")):
        init_embedding_encoder(data_dir, "ent_ids_1")
    if not os.path.exists(os.path.join(data_dir, "ent_ids_2_emb")):
        init_embedding_encoder(data_dir, "ent_ids_2")

    dim = 64

    df = pd.read_table(os.path.join(data_dir, "ent_ids_1_emb"), sep="\t", header=None)
    df_multi = df[1].str.split(",", expand=True).astype(float)
    n_ent1 = len(df_multi)


    df = pd.read_table(os.path.join(data_dir, "ent_ids_2_emb"), sep="\t", header=None)
    data = df[1].str.split(",", expand=True).astype(float)
    n_ent2 = len(data)

    data = pd.concat([df_multi, data],axis=0)
    data = np.array(data.values)

    kernel, bias = compute_kernel_bias(data, dim)
    v_data = transform_and_normalize(data, kernel=kernel, bias = bias)

    ent_1_emb, ent_2_emb = v_data[:n_ent1,:], v_data[n_ent1:,:]
    print(ent_1_emb.shape, ent_2_emb.shape)

    np.savetxt(os.path.join(data_dir, f"ent_1_emb_{dim}.txt"), ent_1_emb)
    np.savetxt(os.path.join(data_dir, f"ent_2_emb_{dim}.txt"), ent_2_emb)

    process_id_features(data_dir, "ent_ids_1", "id_features_1")
    process_id_features(data_dir, "ent_ids_2", "id_features_2")
