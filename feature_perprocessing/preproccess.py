import numpy as np
import json
import argparse 

class Config:
    def __init__(self, data='icews_wiki'):
        self.data = data
        self.path = 'data/' + data + '/'
        self.e1 = self.path + 'id_features_1'
        self.e2 = self.path + 'id_features_2'
        self.r1 = self.path + 'rel_ids_1'
        self.r2 = self.path + 'rel_ids_2'

        self.ill = self.path + 'ref_ent_ids'
        self.kg1 = self.path + 'triples_1'
        self.kg2 = self.path + 'triples_2'

def load_ents(path):
    data = {}
    with open(path,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            data[line[0]] = line[1].replace(' ','')
    print('load %s %d'%(path,len(data)))
    return data
def load_rel(path):
    data = {}
    with open(path,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            data[line[0]] = line[1].split('/')[-1]
    print('load %s %d'%(path,len(data)))
    return data
def load_file(path):
    data = []
    with open(path,'r') as f:
        for line in f:
            line = line.strip().split('\t')[:3]
            data.append(line)
    print('load %s %d'%(path,len(data)))
    return data

def save_data(path,data):
    with open(path,'w') as f:
        n = len(data)
        f.write('%d\n'%n)
        for line in data:
            f.write('%s\n'%'\t'.join([i for i in line]))

def process_specical_word(c):
    rules = [[['à','á','â','ã','ä','å','ā','ă','ạ','ả','ấ','ầ','ẩ','ẫ','ậ','ắ','ằ','ẵ','а'],'a'],
            [['Á','Å'],'A'],
            [['в'],'b'],
            [['ç','ć','Ċ','č','с'],'c'],
            [['Ç','Ć','Č'],'C'],
            [['ð'],'d'],
            [['Đ'],'D'],
            [['Æ','è','é','ê','ë','ė','ē','ę','ě','ế','ề','ễ','ệ'],'e'],
            [['É'],'E'],
            [['ğ'],'g'],
            [['н'],'h'],
            [['ì','í','î','ï','ĩ','ī','ı','ľ','ị','Î'],'i'],
            [['Ď'],'J'],
            [['ķ'],'k'],
            [['ł','İ'],'l'],
            [['Ľ','Ł'],'L'],
            [['ṃ'],'m'],
            [['ñ','ń','ň','ņ'],'n'],
            [['И'],'N'],
            [['ø','ò','ó','ô','õ','ö','ō','ő','ũ','ơ','ọ','ố','ồ','ổ','ộ','ớ','ờ','ở','ợ','о'],'o'],
            [['Ö','Ø','Ó','Ō'],'O'],
            [['р'],'p'],
            [['œ'],'oe'],
            [['ř'],'r'],
            [['Þ','ś','ş','š','ș'],'s'],
            [['Ś','Ş','Š'],'S'],
            [['ß'],'ss'],
            [['ț'],'t'],
            [['ù','ú','ü','ū','ű','ư','ụ','ủ','ứ','ừ','ữ','ự'],'u'],
            [['Ú','Ü'],'U'],
            [['ý','ỹ','ÿ','ỳ'],'y'],
            [['ž'],'z'],
            [['Ž'],'Z'],
            ]
    for rule,replace in rules:
        if c in rule:
            return replace
    return c

def conver_name(name):
    name = ''.join([process_specical_word(i) for i in name])
    return name

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--l', type=str, help='data',default='icews_wiki')
    args = parser.parse_args()
    data = args.l
    config = Config(data)
    ents_1 = load_ents(config.e1)
    ents_2 = load_ents(config.e2)
    rel_1 = load_rel(config.r1)
    rel_2 = load_rel(config.r2)
    ILL = load_file(config.ill)
    kg_1 = load_file(config.kg1)
    kg_2 = load_file(config.kg2)

    ents = ents_1.copy()
    ents.update(ents_2)
    rel = rel_1.copy()
    rel.update(rel_2)
    ents_num = len(ents)
    rel_num = len(rel)
    kg = kg_1 + kg_2

    train = ILL[:1500]
    test = ILL[1500:]
    same_name = {}
    for id_1,id_2 in train:
        name = id_1+"-"+id_2
        same_name[name] = [id_1,id_2]

    node2same = {}
    c=0
    for name,ids in same_name.items():
        if len(ids) == 2 and ((ids[0] in ents_1 and ids[1] in ents_2) or (ids[0] in ents_2 and ids[1] in ents_1)):
            node2same[ids[0]] = str(ents_num+c)
            node2same[ids[1]] = str(ents_num+c)
            c+=1

    same_rel = {}    
    for id,name in rel.items():
        if name in same_rel:
            same_rel[name].append(id)
        else:
            same_rel[name] = [id]

    rel2same = {}
    c = 0
    for _,ids in same_rel.items():
        if len(ids) > 1:
            for id in ids:
                rel2same[id] = str(rel_num+c)
            c += 1

    start_index = max([int(i) for i in node2same.values()]) + 1
    rel2index={}
    node2rel={}
    c = 0
    for h,r,t in kg:
        if h in node2same:
            h = node2same[h]
        if r in rel2same:
            r = rel2same[r]
        if t in node2same:
            t = node2same[t]
        
        if r in rel2index:
            r = rel2index[r]
        else:
            rel2index[r] = str(start_index + c)
            c += 1
            
            r = rel2index[r]
        node2rel[h + '+' + t] = r
        node2rel[t + '+' + h] = r
    with open(config.path+'node2same','w') as f:
        json.dump(node2same,f)
    with open(config.path+'rel2same','w') as f:
        json.dump(rel2same,f)
    
    with open(config.path+'rel2index','w') as f:
        json.dump(rel2index,f)
    with open(config.path+'node2rel','w') as f:
        json.dump(node2rel,f)
    
    print('same node:%d same rel:%d'%(len(node2same),len(rel2same)))
    # fast data
    with open(config.path+'fast.data','w') as f:
        for h,r,t in kg:
            f.write(ents[h]+' '+ents[t]+'\n')
    # deepwalk data
    with open(config.path+'deepwalk.data','w') as f:
        for h,r,t in kg:
            if h in node2same:
                h = node2same[h]
            if t in node2same:
                t = node2same[t]
            f.write(h + ' '+ t +'\n')
    
    # transE data
    all_ents = ents.copy()
    for _,id in node2same.items():
        all_ents[id] = 'same_node_%s'%id

    all_rel = rel.copy()
    for _,id in rel2same.items():
        all_rel[id] = 'same_rel_%s'%id  

    triples = np.array([ 
        [node2same[h] if h in node2same else h,
        rel2same[r] if r in rel2same else r,
        node2same[t] if t in node2same else t] 
        for h,r,t in kg ])
    all_ents = np.array([[i,name] for i,name in all_ents.items()])
    all_rel = np.array([[i,name] for i,name in all_rel.items()])

    save_data(config.path + 'entity2id.txt',all_ents[:,[1,0]])
    save_data(config.path + 'relation2id.txt',all_rel[:,[1,0]])
    save_data(config.path + 'train2id.txt',triples[:,[0,2,1]])
    
