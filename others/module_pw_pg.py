import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # bnc
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)  # 3b h n c/h
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)         b h n c/h

        attn = (q @ k.transpose(-2, -1)) * self.scale  # b h n n
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)  # bhn c/h -> bnc
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):  # bnc
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x  # bnc


class TransformerEncoder(nn.Module):
    """
    ViT-B:
        depth=12, head=12;
    ViT-L:
        depth=24, head=16;
    ViT-H:
        depth=32, head=16;
    """
    def __init__(self, depth=12, embed_dim=512, num_head=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0,
                 attn_drop_rate=0, drop_path_rate=0., output_slice=True):
        super(TransformerEncoder, self).__init__()
        self.slice = output_slice
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_head, mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                           attn_drop=attn_drop_rate,
                                           drop_path=drop_path_rate[i]
                                           if isinstance(drop_path_rate, list) else drop_path_rate)
                                     for i in range(depth)])

    def forward(self, final_input):
        tmp = final_input
        for block in self.blocks:
            tmp = block(tmp)
        if self.slice:
            cls_token = tmp[:, 0, :].unsqueeze(1)  # b 1 c
            representation = tmp[:, 1:, :]  # b seq_len c
            return cls_token, representation
        else:
            return tmp  # b seq_len+1 c


class Embedding(nn.Module):
    def __init__(self, embedding_dim=512, group=64, pos=True):
        super(Embedding, self).__init__()
        self.pos = pos
        self.patch_embed_1 = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.patch_embed_2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, embedding_dim, 1),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embedding_dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, group, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embedding_dim))
        trunc_normal_(self.pos_embedding, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def forward(self, x):
        b, _, seq_len, group, c = x.shape  # b2ng3
        pt_1 = x[:, 0, :, :, :].squeeze().reshape(b*seq_len, group, c)  # b*n g 3
        pt_2 = x[:, 1, :, :, :].squeeze().reshape(b*seq_len, group, c)  # b*n g 3

        pt1 = self.patch_embed_1(pt_1.transpose(2, 1))  # bn c(256) g
        pt2 = self.patch_embed_1(pt_2.transpose(2, 1))  # bn c(256) g
        pt1g = torch.max(pt1, dim=2, keepdim=True)[0]  # bn c(256) 1
        pt2g = torch.max(pt2, dim=2, keepdim=True)[0]  # bn c(256) 1
        pt1 = self.patch_embed_2(torch.cat([pt1g.expand(-1, -1, group), pt1], dim=1))  # bn c(512) g -> bn c(512) g
        pt2 = self.patch_embed_2(torch.cat([pt2g.expand(-1, -1, group), pt2], dim=1))  # bn c(512) g -> bn c(512) g
        pt1 = torch.max(pt1, dim=2, keepdim=True)[0].reshape(b, seq_len, -1)  # bn c(512) -> b n c
        pt2 = torch.max(pt2, dim=2, keepdim=True)[0].reshape(b, seq_len, -1)  # bn c(512) -> b n c

        cls_token = self.cls_token.expand(b, -1, -1)
        pt1, pt2 = torch.cat((pt1, cls_token), dim=1), torch.cat((pt2, cls_token), dim=1)
        if self.pos:
            pos1, pos2 = self.pos_embedding.expand(b, -1, -1), self.pos_embedding.expand(b, -1, -1)
            cls_pos = self.cls_pos.expand(b, -1, -1)

            pos1, pos2 = torch.cat((pos1, cls_pos), dim=1), torch.cat((pos2, cls_pos), dim=1)
            return pt1+pos1, pt2+pos2  # (b n+1(513) 512) (b n+1(513) 512)
        else:
            return pt1, pt2  # (b n+1(513) 512) (b n+1(513) 512)


class EmbeddingPosself(nn.Module):
    def __init__(self, embedding_dim=512, pos=True):
        super(EmbeddingPosself, self).__init__()
        self.pos = pos
        self.patch_embed_1 = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.patch_embed_2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, embedding_dim, 1),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embedding_dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embedding_dim))
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def forward(self, x, pos):
        pos_1, pos_2 = pos[:, 0, :, :].squeeze(), pos[:, 1, :, :].squeeze()
        b, _, seq_len, group, c = x.shape  # b2ng3
        pt_1 = x[:, 0, :, :, :].squeeze().reshape(b*seq_len, group, c)  # b*n g 3
        pt_2 = x[:, 1, :, :, :].squeeze().reshape(b*seq_len, group, c)  # b*n g 3

        pt1 = self.patch_embed_1(pt_1.transpose(2, 1))  # bn c(256) g
        pt2 = self.patch_embed_1(pt_2.transpose(2, 1))  # bn c(256) g
        pt1g = torch.max(pt1, dim=2, keepdim=True)[0]  # bn c(256) 1
        pt2g = torch.max(pt2, dim=2, keepdim=True)[0]  # bn c(256) 1
        pt1 = self.patch_embed_2(torch.cat([pt1g.expand(-1, -1, group), pt1], dim=1))  # bn c(512) g->bn c(512) g
        pt2 = self.patch_embed_2(torch.cat([pt2g.expand(-1, -1, group), pt2], dim=1))  # bn c(512) g->bn c(512) g
        pt1 = torch.max(pt1, dim=2, keepdim=True)[0].reshape(b, seq_len, -1)  # bn c(512) -> b n c
        pt2 = torch.max(pt2, dim=2, keepdim=True)[0].reshape(b, seq_len, -1)  # bn c(512) -> b n c

        cls_token = self.cls_token.expand(b, -1, -1)
        pt1, pt2 = torch.cat((pt1, cls_token), dim=1), torch.cat((pt2, cls_token), dim=1)
        if self.pos:
            pos1, pos2 = self.pos_embed(pos_1), self.pos_embed(pos_2)
            cls_pos = self.cls_pos.expand(b, -1, -1)

            pos1, pos2 = torch.cat((pos1, cls_pos), dim=1), torch.cat((pos2, cls_pos), dim=1)
            return pt1+pos1, pt2+pos2  # (b n+1(513) 512) (b n+1(513) 512)
        else:
            return pt1, pt2  # (b n+1(513) 512) (b n+1(513) 512)
