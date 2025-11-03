import torch
import torch.nn as nn



class SelfAttention(nn.Module):
    def __init__(self, D=768, num_head=12):
        super().__init__()
        self.D = D
        self.num_head = num_head
        self.head_dim = D // num_head
        self.qk_scale = 1 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        
        self.Q_K_V = nn.Linear(in_features=D, out_features=D * 3)
        self.projection = nn.Linear(in_features=D, out_features=D)

    def forward(self, patch_embeddings):
        B, N, _ = patch_embeddings.shape
        Q_K_V = self.Q_K_V(patch_embeddings)


        Q = Q_K_V[:, :, :self.D]                    
        K = Q_K_V[:, :, self.D:2*self.D]           
        V = Q_K_V[:, :, 2*self.D:3*self.D]         

        
        Q = Q.reshape(B, N, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(B, N, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, N, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        
        attention = torch.matmul(Q, K.transpose(-1, -2)) * self.qk_scale
        probs = torch.softmax(attention, dim=-1)
        x = torch.matmul(probs, V).transpose(1, 2).reshape(B, N, self.D)
        x = self.projection(x)
        return x


class MutilHeadAttention(nn.Module):
    def __init__(self,embed_dim=768,num_head=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.SelfAttention = SelfAttention(D=embed_dim,num_head=num_head)
    def forward(self,patch_embeddings):
        return self.SelfAttention(patch_embeddings)

MSA = MutilHeadAttention(embed_dim=768,num_head=12)


class Encoder(nn.Module):
    def __init__(self,embed_dim=768,num_head=12,mlp_ratio=4.0,dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.LayerNorm1 = nn.LayerNorm(embed_dim)
        self.MHSA = MutilHeadAttention(embed_dim,num_head)
        self.LayerNorm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.MLP = nn.Sequential(nn.LayerNorm(embed_dim)
                                ,nn.Linear(in_features=embed_dim,out_features=hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=hidden_dim,out_features=embed_dim),
                                 nn.Dropout(dropout))
    def forward(self,patch_embeddings):
        x = self.LayerNorm1(patch_embeddings)
        x_MHSA = self.MHSA(x)
        res1 = patch_embeddings + x_MHSA #residual connection 1
        x = self.LayerNorm2(res1)
        x = self.MLP(x)
        out = res1 + x  # residual connection 2
        return out


class ViT(nn.Module):
    def __init__(self,embed_dim=768,num_encoders=12,hidden_dim=3072,num_head=12,mlp_ratio=4.0,dropout=0.1,num_classes=1000):
        super().__init__()
        self.num_encoders = num_encoders
        self.Encoders = nn.ModuleList([Encoder(embed_dim,num_head,mlp_ratio,dropout) for _ in range(num_encoders)])
        self.MLP_head = nn.Linear(in_features=embed_dim,out_features=num_classes)
    def forward(self,patch_embeddings):
        x = patch_embeddings
        Encoders_output = []
        for encoder in self.Encoders:
            x = encoder(x)
            Encoders_output.append(x)
        out = self.MLP_head(x[:,0])
        return out