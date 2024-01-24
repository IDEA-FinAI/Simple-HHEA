import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


### Time2Vec Model
class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features, activation="cos"):
        super(CosineActivation, self).__init__()
        self.w0 = Parameter(torch.randn(in_features, 1))
        self.b0 = Parameter(torch.randn(1))
        self.w  = Parameter(torch.randn(in_features, out_features-1))
        self.b  = Parameter(torch.randn(out_features-1))
        self.f  = torch.cos if activation == "cos" else torch.sin

    def forward(self, tau):
        v1 = self.f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], 1)


class Time2Vec(nn.Module):
    def __init__(self, hidden_dim):
        super(Time2Vec, self).__init__()
        self.l = CosineActivation(1, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        x = self.l(x)
        x = self.fc(x)
        return x


### Main Model, Simple_HHEA
class Simple_HHEA(nn.Module):
    def __init__(self, time_span, ent_name_emb, ent_time_emb, ent_dw_emb, use_structure=True, use_time=True, emb_size=64, structure_size=8, time_size=8, device="cuda"):
        super(Simple_HHEA, self).__init__()

        self.device = device
        self.use_structure = use_structure
        self.use_time = use_time

        self.emb_size = emb_size
        self.struct_size = structure_size
        self.time_size = time_size

        linear_size_1 = self.emb_size

        if self.use_time:
            linear_size_1 += self.time_size
            self.ent_time_emb = torch.tensor(ent_time_emb).to(self.device).float()
            self.fc_time_0 = nn.Linear(32, 32)
            self.fc_time = nn.Linear(32, self.time_size)
            self.time2vec = Time2Vec(hidden_dim=32)
            self.time_span = time_span 
            self.time_span_index = torch.tensor(np.array([i for i in range(self.time_span)])).to(self.device).unsqueeze(1).float()

        if self.use_structure:
            linear_size_1 += self.struct_size
            self.ent_dw_emb = torch.tensor(ent_dw_emb).to(self.device).float()
            self.fc_dw_0 = nn.Linear(self.ent_dw_emb.shape[-1], emb_size)
            self.fc_dw = nn.Linear(emb_size, self.struct_size)
        
        self.fc_final = nn.Linear(linear_size_1, emb_size)

        self.ent_name_emb = torch.tensor(ent_name_emb).to(self.device).float()
        self.fc_name_0 = nn.Linear(self.ent_name_emb.shape[-1], emb_size)
        self.fc_name = nn.Linear(emb_size, emb_size)

        self.dropout = nn.Dropout(p=0.3)
        self.activation = nn.ReLU()

    def forward(self):
        ent_name_feature = self.fc_name(self.fc_name_0(self.dropout(self.ent_name_emb)))
        features = [ent_name_feature]

        if self.use_time:
            time_span_feature = self.time2vec(self.time_span_index)
            ent_time_feature = torch.mm(self.ent_time_emb, time_span_feature) / self.time_span
            ent_time_feature = self.fc_time(self.fc_time_0(self.dropout(ent_time_feature)))
            features.append(ent_time_feature)
        
        if self.use_structure:
            ent_dw_feature = self.fc_dw(self.fc_dw_0(self.dropout(self.ent_dw_emb)))
            features.append(ent_dw_feature)

        output_feature = torch.cat(features, dim=1)
        output_feature = self.fc_final(output_feature)
        
        return output_feature