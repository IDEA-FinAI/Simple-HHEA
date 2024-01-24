import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch

from CSLS_ import eval_alignment_by_sim_mat
from model import Simple_HHEA
from utils import *


### load embeddings
def noise_name_emb(name_emb, noise_ratio, emb_size=64):
    sample_list = [i for i in range(emb_size)]
    mask_id = random.sample(sample_list, int(emb_size * noise_ratio))
    name_emb[:, mask_id] = 0
    return name_emb

def load_embeddings(data_path, add_noise, noise_ratio, use_structure=True, use_time=True):
    ent_name_emb, ent_dw_emb, ent_time_emb = None, None, None
    ### load name embeddings
    kg1_name_emb = np.loadtxt(os.path.join(data_path, "ent_1_emb_64.txt"))
    kg2_name_emb = np.loadtxt(os.path.join(data_path, "ent_2_emb_64.txt"))
    ent_name_emb = np.array(kg1_name_emb.tolist() + kg2_name_emb.tolist())
    print(f"read entity name embedding shape: {ent_name_emb.shape}")
    if add_noise:
        ent_name_emb = noise_name_emb(ent_name_emb, noise_ratio)
    ### load structure embeddings
    if use_structure:
        ent_dw_emb = np.loadtxt(os.path.join(data_path, "deep_emb.txt"))
        print(f"read entity deepwalk emb shape: {ent_dw_emb.shape}")
    ### load time embeddings
    if use_time:
        ent_time_emb = np.array(load_ent_time_matrix(data_path))
        print(f"read entity time embedding shape: {ent_time_emb.shape}")
    return ent_name_emb, ent_dw_emb, ent_time_emb


### training
def l1(ll, rr):
    return torch.sum(torch.abs(ll - rr), axis=-1)

def evaluate(model, dev_alignments, hit_k=[1, 5, 10], num_threads=16, csls=10):
    model.eval()
    with torch.no_grad():
        feat = model()[dev_alignments]
        Lvec, Rvec = feat[:, 0, :].detach().cpu().numpy(), feat[:, 1, :].detach().cpu().numpy()
        Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        t_prec_set, acc, t_mrr = eval_alignment_by_sim_mat(Lvec, Rvec, hit_k, num_threads, csls, accurate=True)
        acc = [round(n, 3) for n in acc]
    return t_prec_set, acc, t_mrr

def train(model:nn.Module, alignment_pairs, dev_alignments, epochs=1500, learning_rate=0.01, weight_decay=0.001, gamma=1.0, hit_k=[1, 5, 10]):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print(f"parameters: {get_n_params(model)}")

    losses = []
    t_prec = []
    accs = []
    t_mrrs = []
    best_acc = [0] * len(hit_k)
    best_mrr = 0
    batch_size = len(alignment_pairs)
    for i in tqdm(range(epochs)):
        ### forwad
        model.train()
        optimizer.zero_grad()
        feat = model()[alignment_pairs]
        ### loss
        l, r, fl, fr = feat[:, 0, :], feat[:, 1, :], feat[:, 2, :], feat[:, 3, :]
        loss = torch.sum(nn.ReLU()(gamma + l1(l, r) - l1(l, fr)) + nn.ReLU()(gamma + l1(l, r) - l1(fl, r))) / batch_size
        ### backward
        losses.append(loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()
        ### evaluate
        if (i + 1) % 10 == 0:
            t_prec_set, acc, t_mrr = evaluate(model, dev_alignments, hit_k)

            for i in range(len(hit_k)):
                if best_acc[i] < acc[i]:
                    best_acc[i] = acc[i]
            if best_mrr < t_mrr:
                best_mrr = t_mrr
            print(f"//best results: hits@{hit_k} = {best_acc}, mrr = {best_mrr:.3f}//")
            accs.append(acc)
            t_mrrs.append(t_mrr)
            t_prec.append(t_prec_set)
    return losses, t_prec, accs, t_mrrs, best_acc, best_mrr




if __name__ == "__main__":
    ### hyper parmeters
    parser = argparse.ArgumentParser(description="Simple-HHEA Experiment")
    parser.add_argument("--data", type=str, default="icews_wiki")
    parser.add_argument("--cuda", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=12306)
    ###### ablation settings
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--noise_ratio", type=float, default=0.3)
    parser.add_argument("--no_structure", action="store_true")
    parser.add_argument("--no_time", action="store_true")
    ###### training settings
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1500)
    args = parser.parse_args()
    
    ### basic settings
    data = args.data
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    use_time = ("icews_wiki" in data or "icews_yago" in data) and not args.no_time
    use_structure = not args.no_structure
    print(f"start exp: noise_ratio={args.noise_ratio}, data=\"{args.data}\", use_structure={use_structure}, use_time={use_time}")
    ### random settings
    fixed(args.random_seed)

    ### load datas
    data_path = os.path.join("data", data)
    all_triples, node_size, rel_size = load_triples(data_path, True)
    print(f"node_size={node_size} , rel_size={rel_size}")

    train_alignments = load_alignments(os.path.join(data_path, "sup_pairs"))
    dev_alignments = load_alignments(os.path.join(data_path, "ref_pairs"))
    print(f"Train/Val: {len(train_alignments)}/{len(dev_alignments)}")

    ### load name embeddings
    ent_name_emb, ent_dw_emb, ent_time_emb = load_embeddings(data_path, args.add_noise, args.noise_ratio, use_structure, use_time)

    ### model
    model = Simple_HHEA(
        time_span=1+27*13,
        ent_name_emb=ent_name_emb,
        ent_time_emb=ent_time_emb,
        ent_dw_emb=ent_dw_emb,
        use_structure=use_structure,
        use_time=use_time,
        emb_size=64,
        structure_size=8,
        time_size=8,
        device=device
    )
    model = model.to(device)

    alignment_pairs = get_train_set(train_alignments, node_size, node_size)
    losses, t_prec, accs, t_mrrs, best_acc, best_mrr = train(model, alignment_pairs, dev_alignments, args.epochs, args.lr, args.wd, args.gamma, hit_k=[1, 5, 10])

    ### save result
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, f"{data}_result_file_mlp.txt"), "a+", encoding="utf-8") as fw:
        fw.write(f"settings: noise_ratio: {args.noise_ratio}, use_time: {use_time}, use_structure: {use_structure}\n\tbest results: hits@[1, 5, 10] = {best_acc}, mrr = {best_mrr:.3f}\n")