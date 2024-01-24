import json
import numpy as np
import torch
import torch.nn.functional as F
import gensim
import argparse 
from preproccess import load_ents,load_file

def get_node_vec(wv, node2same,ents,wv_type='fast'):
    node_vec={}
    if  wv_type == 'fast':
        for id, name in ents.items():
            try:
                node_vec[id] = wv[name]
            except:
                node_vec[id] = np.random.random_sample(size = (1, 64))[0]
    if wv_type == 'deep':
        for id in ents:
            new_id = id
            if id in node2same:
                new_id = node2same[id]
            try:
                node_vec[id] = wv[new_id]
            except:
                node_vec[id] = np.random.random_sample(size = (1, 64))[0]
    
    vec = []
    for i in range(len(ents)):
        vec.append(node_vec[str(i)])
    vec = torch.tensor(np.array(vec),requires_grad=False)
    print(vec.shape)
    return vec

def get_node_context(node_v,rel_v,ents,kg):
    node_context = {}
    for h,r,t in kg:
        if h not in node_context:
            node_context[h] = []
        if t not in node_context:
            node_context[t] = []
        node_context[h].append(node_v[t]-rel_v[r])
        node_context[t].append(node_v[h]+rel_v[r])
    for node in node_context:
        node_context[node] = np.mean(node_context[node],axis=0)
    vec = []
    for id in range(len(ents)): 
        id = str(id)
        if id in node2same:
            id = node2same[id]
        #TODO
        try:
            vec.append(node_context[id])
        except:
            vec.append(np.random.random_sample(size = (1, 64))[0])
    vec = torch.tensor(vec,requires_grad=False)
    print(vec.shape)	
    return vec

def eval_embd(ent_vecs, test_pair, device, top_k=(1, 10)):
    ent_vecs = ent_vecs.to(device)
    with torch.no_grad():
        test_left = torch.LongTensor(test_pair[:, 0].squeeze()).to(device)
        test_right = torch.LongTensor(test_pair[:, 1].squeeze()).to(device)
        test_left_vec = ent_vecs[test_left]
        test_right_vec = ent_vecs[test_right]
        he = F.normalize(test_left_vec, p=2, dim=-1)
        norm_e_em = F.normalize(test_right_vec, p=2, dim=-1)
        sim = torch.matmul(he, norm_e_em.t())
        cos_distance = 1 - sim
        
        top_lr = [0] * len(top_k)
        mrr_sum_l = 0
        ranks = torch.argsort(cos_distance)
        rank_indexs = torch.where(ranks == torch.arange(len(test_pair)).unsqueeze(1).to(device))[1].cpu().numpy().tolist()

        for i in range(cos_distance.shape[0]):
            rank_index = rank_indexs[i] 
            mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_lr[j] += 1  
            
        msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (
            top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair))
        print(msg)

    return  top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair)

def get_embedding_by_pair(embedding,pair,device):
    with torch.no_grad():
        left = torch.LongTensor(pair[:, 0].squeeze())
        right = torch.LongTensor(pair[:, 1].squeeze())
    return F.normalize(embedding[left],p=2,dim=1).detach_().to(device),F.normalize(embedding[right],p=2,dim=1).detach_().to(device)

def stable_match(sim):
    softmax_sim = torch.softmax(torch.from_numpy(sim),-1).numpy()+torch.softmax(torch.from_numpy(sim),0).numpy()
    sim_shape = softmax_sim.shape
    softmax_sim = np.reshape(softmax_sim,(1,-1))
    sort_sim = np.argsort(-softmax_sim)

    x,y = sim_shape
    result={}
    book = set([])

    for _,i in enumerate(sort_sim.tolist()[0]):
        if len(result.keys()) == sim_shape[0]:
            break
        a = i // y
        b = i % y + x
        if a in book:
            continue
        if b in book:
            continue
        result[a] = b
        book.add(a)
        book.add(b)
        
    c = 0
    for a,b in result.items():
        if a + x ==b:
            c += 1
    return c/len(result.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,default='data/icews_wiki/')
    parser.add_argument('--output', type=str,default='result/final.txt')
    args = parser.parse_args()

    path = args.input
    output = args.output

    ent_1 = load_ents(path+'id_features_1')
    print('load ent_1',len(ent_1))
    ent_2 = load_ents(path+'id_features_2')
    print('load ent_2',len(ent_2))
    ents = ent_1.copy()
    ents.update(ent_2)
    print('load ents',len(ents))

    ILL = load_file(path+'ref_ent_ids')
    ILL = [[int(i),int(j)] for i,j in ILL]
    ILL = np.array(ILL)
    pair_num = len(ILL)
    train_ILL = ILL[:int(pair_num*0.3)]
    test_ILL = ILL[int(pair_num*0.3):]

    kg_1 = load_file(path+'triples_1')
    kg_2 = load_file(path+'triples_2')
    kg = kg_1 + kg_2
    print('load kg',len(kg))

    fast_wv = gensim.models.KeyedVectors.load_word2vec_format(path+'fast.vec')
    deep_wv = gensim.models.KeyedVectors.load_word2vec_format(path+'longterm.vec')
    transe_wv = gensim.models.KeyedVectors.load_word2vec_format(path+'transe.vec')
    transe_rel = gensim.models.KeyedVectors.load_word2vec_format(path+'transe_rel.vec')

    with open(path+'node2same','r') as f:
        node2same = json.load(f)

    with open(path+'rel2same','r') as f:
        rel2same = json.load(f)
    
    print('node2same:',len(node2same),'rel2same:',len(rel2same))
    # get merge kg
    M_KG = []
    for h,r,t in kg:
        if h in node2same:
            h = node2same[h]
        if t in node2same:
            t = node2same[t]
        if r in rel2same:
            r = rel2same[r]
        M_KG.append([h,r,t])
    

    fast_node = get_node_vec(fast_wv,node2same,ents,'fast')
    deep_node = get_node_vec(deep_wv,node2same,ents,'deep')
    transe_node = get_node_vec(transe_wv,node2same,ents,'deep')
    context_node = get_node_context(transe_wv,transe_rel,ents,M_KG)

    deep_emb = deep_node.numpy()
    transe_emb = transe_node.numpy()
    context_emb = context_node.numpy()

    # np.savetxt(path + "deep_emb.txt",deep_emb)
    # np.savetxt(path + "transe_emb.txt",transe_emb)
    # np.savetxt(path + "context_emb.txt",context_emb)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    
    y = 0.5
    s_1 = torch.cat((
                        F.normalize(y*transe_node + (1-y)*context_node,dim=-1),
                        F.normalize(deep_node,dim=-1),
                        F.normalize(fast_node,dim=-1)
                    ), dim=1)
    s_1 = F.normalize(s_1, dim=-1)
    s_list = [s_1]

    result = []
    for s in s_list:
        t1,t10,mrr = eval_embd(s, test_ILL, device)

        train_left,train_right = get_embedding_by_pair(s,train_ILL,device)
        test_left,test_right = get_embedding_by_pair(s,test_ILL,device)
        #RSM
        left_norm = F.normalize(test_left, p=2, dim=-1)
        right_norm = F.normalize(test_right, p=2, dim=-1)
        sim = torch.matmul(left_norm, right_norm.t()).cpu().detach().numpy()
        my_stab = stable_match(sim)
        print("Reliability-based Stable Matching:",my_stab)
        result.append((t1,t10,mrr,my_stab))

    with open(output,'a') as f:
        f.write(path+'\n')
        for r in result:
            f.write('%.4f %.4f %.4f %.4f\n'%r)







    